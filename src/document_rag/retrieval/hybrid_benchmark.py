"""Run the controlled BM25 and dense reciprocal-rank fusion benchmark."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

from document_rag.datasets.models import DatasetExample, DatasetName, DatasetSplit
from document_rag.retrieval.benchmark import (
    BenchmarkDatasetCounts,
    FrozenBenchmarkCorpus,
    build_metrics_by_dataset,
    load_frozen_benchmark_corpus,
    serialize_evaluations_jsonl,
    write_bytes_atomically,
)
from document_rag.retrieval.bm25 import (
    TOKENIZATION_STRATEGY,
    BM25Retriever,
)
from document_rag.retrieval.dense import (
    DenseEmbedder,
    DenseRetriever,
    FloatMatrix,
    FloatVector,
    normalize_embedding_matrix,
)
from document_rag.retrieval.embedding_cache import DocumentEmbeddingCache
from document_rag.retrieval.embeddings import (
    BGE_QUERY_PROMPT,
    DOCUMENT_ENCODING_STRATEGY,
    QUERY_ENCODING_STRATEGY,
)
from document_rag.retrieval.evaluation import (
    DEFAULT_K_VALUES,
    RetrievalEvaluationError,
    aggregate_evaluations,
    evaluate_query,
    normalize_k_values,
)
from document_rag.retrieval.hybrid import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_RRF_K,
    RRF_FORMULA,
    fuse_rankings,
)
from document_rag.retrieval.models import (
    QueryRetrievalEvaluation,
    RetrievalMetrics,
)

HYBRID_PREDICTIONS_FILENAME = "hybrid_predictions.jsonl"
HYBRID_METRICS_FILENAME = "hybrid_metrics.json"
COMPARISON_FILENAME = "retrieval_comparison.json"


@dataclass(frozen=True, slots=True)
class HybridBenchmarkResult:
    """Artifacts and metrics from one controlled reciprocal-rank fusion run."""

    predictions_path: Path
    metrics_path: Path
    comparison_path: Path
    embeddings_path: Path
    embedding_manifest_path: Path
    embedding_cache_reused: bool
    evaluations: tuple[QueryRetrievalEvaluation, ...]
    metrics: RetrievalMetrics
    dataset_counts: tuple[tuple[DatasetName, BenchmarkDatasetCounts], ...]
    indexed_chunk_count: int
    document_count: int
    corpus_sha256: str
    query_set_sha256: str


def run_hybrid_benchmark(
    *,
    dataset_directories: Mapping[DatasetName, Path],
    output_directory: Path,
    bm25_metrics_path: Path,
    dense_metrics_path: Path,
    embedder: DenseEmbedder,
    rrf_k: int = DEFAULT_RRF_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    batch_size: int = 32,
    split: DatasetSplit = DatasetSplit.TEST,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
    cache_directory: Path | None = None,
) -> HybridBenchmarkResult:
    """Evaluate deterministic RRF on the exact frozen lexical/dense inputs."""

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive.")

    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    selected_k_values = normalize_k_values(k_values)
    corpus = load_frozen_benchmark_corpus(
        dataset_directories=dataset_directories,
        split=split,
    )
    output_root = output_directory.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = (
        cache_directory.resolve()
        if cache_directory is not None
        else output_root / "embedding_cache"
    )
    cached_embeddings = DocumentEmbeddingCache(cache_root).load_or_encode(
        chunks=corpus.chunks,
        embedder=embedder,
        batch_size=batch_size,
        corpus_sha256=corpus.corpus_sha256,
    )
    query_embeddings = normalize_embedding_matrix(
        embedder.embed_queries(
            tuple(example.question.text for example in corpus.examples),
            batch_size=batch_size,
        ),
        expected_rows=len(corpus.examples),
        expected_dimension=embedder.dimension,
        label="Query",
    )
    bm25_by_document, dense_by_document = _build_component_retrievers(
        corpus,
        embeddings=cached_embeddings.embeddings,
        embedder=embedder,
        batch_size=batch_size,
    )
    evaluations = tuple(
        sorted(
            (
                _evaluate_hybrid_example(
                    example=example,
                    query_embedding=query_embeddings[index],
                    bm25_by_document=bm25_by_document,
                    dense_by_document=dense_by_document,
                    rrf_k=rrf_k,
                    candidate_k=candidate_k,
                    k_values=selected_k_values,
                )
                for index, example in enumerate(corpus.examples)
            ),
            key=lambda evaluation: (
                evaluation.dataset.value,
                evaluation.example_id,
            ),
        )
    )
    metrics = aggregate_evaluations(evaluations)
    metrics_payload = _build_hybrid_metrics_payload(
        corpus=corpus,
        evaluations=evaluations,
        metrics=metrics,
        embedder=embedder,
        rrf_k=rrf_k,
        candidate_k=candidate_k,
        batch_size=batch_size,
        split=split,
        k_values=selected_k_values,
    )
    comparison_payload = _build_comparison_payload(
        bm25_metrics_path=bm25_metrics_path.resolve(),
        dense_metrics_path=dense_metrics_path.resolve(),
        hybrid_metrics_payload=metrics_payload,
    )
    predictions_path = output_root / HYBRID_PREDICTIONS_FILENAME
    metrics_path = output_root / HYBRID_METRICS_FILENAME
    comparison_path = output_root / COMPARISON_FILENAME

    write_bytes_atomically(
        predictions_path,
        serialize_evaluations_jsonl(evaluations),
    )
    write_bytes_atomically(metrics_path, _serialize_json(metrics_payload))
    write_bytes_atomically(comparison_path, _serialize_json(comparison_payload))

    return HybridBenchmarkResult(
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        comparison_path=comparison_path,
        embeddings_path=cached_embeddings.embeddings_path,
        embedding_manifest_path=cached_embeddings.manifest_path,
        embedding_cache_reused=cached_embeddings.reused,
        evaluations=evaluations,
        metrics=metrics,
        dataset_counts=corpus.dataset_counts,
        indexed_chunk_count=corpus.indexed_chunk_count,
        document_count=corpus.document_count,
        corpus_sha256=corpus.corpus_sha256,
        query_set_sha256=corpus.query_set_sha256,
    )


def _build_component_retrievers(
    corpus: FrozenBenchmarkCorpus,
    *,
    embeddings: FloatMatrix,
    embedder: DenseEmbedder,
    batch_size: int,
) -> tuple[dict[str, BM25Retriever], dict[str, DenseRetriever]]:
    bm25_by_document: dict[str, BM25Retriever] = {}
    dense_by_document: dict[str, DenseRetriever] = {}
    row_start = 0

    for dataset in corpus.datasets:
        for document_id, chunks in dataset.chunks_by_document:
            row_end = row_start + len(chunks)
            bm25_by_document[document_id] = BM25Retriever(chunks)
            dense_by_document[document_id] = DenseRetriever(
                chunks,
                embedder=embedder,
                batch_size=batch_size,
                document_embeddings=embeddings[row_start:row_end],
            )
            row_start = row_end

    if row_start != corpus.indexed_chunk_count:
        raise ValueError("Embedding rows do not align with the frozen chunk corpus.")

    return bm25_by_document, dense_by_document


def _evaluate_hybrid_example(
    *,
    example: DatasetExample,
    query_embedding: FloatVector,
    bm25_by_document: Mapping[str, BM25Retriever],
    dense_by_document: Mapping[str, DenseRetriever],
    rrf_k: int,
    candidate_k: int,
    k_values: tuple[int, ...],
) -> QueryRetrievalEvaluation:
    document_id = example.question.document_id

    try:
        bm25 = bm25_by_document[document_id]
        dense = dense_by_document[document_id]
    except KeyError as error:
        raise RetrievalEvaluationError(
            f"No hybrid component index exists for example {example.example_id!r}."
        ) from error

    results = fuse_rankings(
        bm25_results=bm25.search(
            example.question.text,
            top_k=candidate_k,
        ),
        dense_results=dense.search_vector(
            query_embedding,
            top_k=candidate_k,
        ),
        rrf_k=rrf_k,
        top_k=max(k_values),
    )
    return evaluate_query(
        dataset=example.dataset,
        example_id=example.example_id,
        question=example.question.text,
        gold_source_element_ids=(fact.element_id for fact in example.supporting_facts),
        results=results,
        k_values=k_values,
    )


def _build_hybrid_metrics_payload(
    *,
    corpus: FrozenBenchmarkCorpus,
    evaluations: tuple[QueryRetrievalEvaluation, ...],
    metrics: RetrievalMetrics,
    embedder: DenseEmbedder,
    rrf_k: int,
    candidate_k: int,
    batch_size: int,
    split: DatasetSplit,
    k_values: tuple[int, ...],
) -> dict[str, object]:
    return {
        "benchmark_config": {
            "batch_size": batch_size,
            "bm25_configuration_identity": {
                "retriever_type": "bm25_okapi",
                "tokenization_strategy": TOKENIZATION_STRATEGY,
            },
            "candidate_k": candidate_k,
            "dense_embedding_dimension": embedder.dimension,
            "dense_embedding_model": embedder.model_id,
            "dense_embedding_model_revision": embedder.model_revision,
            "dense_normalization": "l2_unit",
            "dense_query_encoding_strategy": QUERY_ENCODING_STRATEGY,
            "dense_query_prompt": BGE_QUERY_PROMPT,
            "dense_similarity_metric": "cosine_via_dot_product",
            "device": embedder.device,
            "document_encoding_strategy": DOCUMENT_ENCODING_STRATEGY,
            "evaluation_top_k": list(k_values),
            "fusion_inputs": "component_ranks_only",
            "k_values": list(k_values),
            "retrieval_scope": "question_document",
            "retriever_type": "hybrid_rrf",
            "rrf_formula": RRF_FORMULA,
            "rrf_k": rrf_k,
            "split": split.value,
            "tie_breaker": "rrf_score_descending_then_chunk_id_ascending",
        },
        "benchmark_identity": {
            "corpus_sha256": corpus.corpus_sha256,
            "query_set_sha256": corpus.query_set_sha256,
        },
        "dataset_counts": {
            dataset.value: counts.to_record() for dataset, counts in corpus.dataset_counts
        },
        "document_count": corpus.document_count,
        "indexed_chunk_count": corpus.indexed_chunk_count,
        "metrics": metrics.to_record(),
        "metrics_by_dataset": build_metrics_by_dataset(
            evaluations=evaluations,
            dataset_counts=corpus.dataset_counts,
        ),
        "query_count": len(evaluations),
    }


def _build_comparison_payload(
    *,
    bm25_metrics_path: Path,
    dense_metrics_path: Path,
    hybrid_metrics_payload: dict[str, object],
) -> dict[str, object]:
    bm25_payload = _load_json_object(bm25_metrics_path, label="BM25 metrics")
    dense_payload = _load_json_object(dense_metrics_path, label="dense metrics")
    _validate_comparable_benchmarks(
        bm25_payload=bm25_payload,
        dense_payload=dense_payload,
        hybrid_payload=hybrid_metrics_payload,
    )
    bm25_by_scope = _metrics_by_scope(bm25_payload)
    dense_by_scope = _metrics_by_scope(dense_payload)
    hybrid_by_scope = _metrics_by_scope(hybrid_metrics_payload)

    return {
        "baseline_artifacts": {
            "bm25_metrics_sha256": sha256(bm25_metrics_path.read_bytes()).hexdigest(),
            "dense_metrics_sha256": sha256(dense_metrics_path.read_bytes()).hexdigest(),
        },
        "benchmark_identity": hybrid_metrics_payload["benchmark_identity"],
        "delta_definitions": (
            "hybrid_minus_bm25",
            "hybrid_minus_dense",
        ),
        "results": {
            scope: {
                "absolute_delta": {
                    "hybrid_minus_bm25": _metric_delta(
                        baseline=cast(dict[str, object], bm25_by_scope[scope]),
                        candidate=cast(dict[str, object], hybrid_by_scope[scope]),
                        baseline_label="BM25",
                    ),
                    "hybrid_minus_dense": _metric_delta(
                        baseline=cast(dict[str, object], dense_by_scope[scope]),
                        candidate=cast(dict[str, object], hybrid_by_scope[scope]),
                        baseline_label="dense",
                    ),
                },
                "bm25": bm25_by_scope[scope],
                "dense": dense_by_scope[scope],
                "hybrid": hybrid_by_scope[scope],
            }
            for scope in sorted(hybrid_by_scope)
        },
    }


def _validate_comparable_benchmarks(
    *,
    bm25_payload: dict[str, object],
    dense_payload: dict[str, object],
    hybrid_payload: dict[str, object],
) -> None:
    for label, payload in (("BM25", bm25_payload), ("dense", dense_payload)):
        for field in (
            "dataset_counts",
            "document_count",
            "indexed_chunk_count",
            "query_count",
        ):
            if payload.get(field) != hybrid_payload.get(field):
                raise ValueError(f"{label} and hybrid benchmark field {field!r} must match.")

        baseline_config = _require_mapping(
            payload.get("benchmark_config"),
            label=f"{label} benchmark_config",
        )
        hybrid_config = _require_mapping(
            hybrid_payload.get("benchmark_config"),
            label="hybrid benchmark_config",
        )

        for field in ("k_values", "retrieval_scope", "split"):
            if baseline_config.get(field) != hybrid_config.get(field):
                raise ValueError(f"{label} and hybrid benchmark config field {field!r} must match.")

    if dense_payload.get("benchmark_identity") != hybrid_payload.get("benchmark_identity"):
        raise ValueError("Dense and hybrid benchmark identities must match.")

    dense_config = _require_mapping(
        dense_payload.get("benchmark_config"),
        label="dense benchmark_config",
    )
    hybrid_config = _require_mapping(
        hybrid_payload.get("benchmark_config"),
        label="hybrid benchmark_config",
    )
    dense_to_hybrid_fields = {
        "embedding_dimension": "dense_embedding_dimension",
        "embedding_model": "dense_embedding_model",
        "embedding_model_revision": "dense_embedding_model_revision",
        "normalization": "dense_normalization",
        "query_encoding_strategy": "dense_query_encoding_strategy",
        "query_prompt": "dense_query_prompt",
        "similarity_metric": "dense_similarity_metric",
    }

    for dense_field, hybrid_field in dense_to_hybrid_fields.items():
        if dense_config.get(dense_field) != hybrid_config.get(hybrid_field):
            raise ValueError(
                f"Dense field {dense_field!r} must match hybrid field {hybrid_field!r}."
            )


def _metrics_by_scope(payload: dict[str, object]) -> dict[str, object]:
    combined = _require_mapping(payload.get("metrics"), label="combined metrics")
    datasets = _require_mapping(
        payload.get("metrics_by_dataset"),
        label="metrics_by_dataset",
    )
    return {"combined": combined, **datasets}


def _metric_delta(
    *,
    baseline: dict[str, object],
    candidate: dict[str, object],
    baseline_label: str,
) -> dict[str, object]:
    baseline_hit = _require_mapping(
        baseline.get("hit_rate_at_k"),
        label=f"{baseline_label} hit rates",
    )
    candidate_hit = _require_mapping(
        candidate.get("hit_rate_at_k"),
        label="hybrid hit rates",
    )
    baseline_recall = _require_mapping(
        baseline.get("recall_at_k"),
        label=f"{baseline_label} recall",
    )
    candidate_recall = _require_mapping(
        candidate.get("recall_at_k"),
        label="hybrid recall",
    )

    return {
        "hit_rate_at_k": {
            key: _require_number(candidate_hit.get(key), label=f"hybrid Hit@{key}")
            - _require_number(
                baseline_hit.get(key),
                label=f"{baseline_label} Hit@{key}",
            )
            for key in sorted(candidate_hit, key=int)
        },
        "mrr": _require_number(candidate.get("mrr"), label="hybrid MRR")
        - _require_number(baseline.get("mrr"), label=f"{baseline_label} MRR"),
        "recall_at_k": {
            key: _require_number(candidate_recall.get(key), label=f"hybrid Recall@{key}")
            - _require_number(
                baseline_recall.get(key),
                label=f"{baseline_label} Recall@{key}",
            )
            for key in sorted(candidate_recall, key=int)
        },
    }


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{label} contains invalid JSON: {path}.") from error

    return _require_mapping(payload, label=label)


def _require_mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")

    return cast(dict[str, object], value)


def _require_number(value: object, *, label: str) -> float:
    if not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric.")

    return float(value)


def _serialize_json(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
