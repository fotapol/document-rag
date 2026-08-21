"""Run dense retrieval on the frozen BM25 corpus and compare metrics."""

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
from document_rag.retrieval.models import (
    QueryRetrievalEvaluation,
    RetrievalMetrics,
)

DENSE_PREDICTIONS_FILENAME = "dense_predictions.jsonl"
DENSE_METRICS_FILENAME = "dense_metrics.json"
COMPARISON_FILENAME = "retrieval_comparison.json"


@dataclass(frozen=True, slots=True)
class DenseBenchmarkResult:
    """Artifacts and metrics from one controlled dense experiment."""

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


def run_dense_benchmark(
    *,
    dataset_directories: Mapping[DatasetName, Path],
    output_directory: Path,
    bm25_metrics_path: Path,
    embedder: DenseEmbedder,
    batch_size: int = 32,
    split: DatasetSplit = DatasetSplit.TEST,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
    cache_directory: Path | None = None,
) -> DenseBenchmarkResult:
    """Embed the frozen corpus, evaluate dense results, and compare to BM25."""

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
    chunks = corpus.chunks
    examples = corpus.examples
    cached_embeddings = DocumentEmbeddingCache(cache_root).load_or_encode(
        chunks=chunks,
        embedder=embedder,
        batch_size=batch_size,
        corpus_sha256=corpus.corpus_sha256,
    )
    query_embeddings = normalize_embedding_matrix(
        embedder.embed_queries(
            tuple(example.question.text for example in examples),
            batch_size=batch_size,
        ),
        expected_rows=len(examples),
        expected_dimension=embedder.dimension,
        label="Query",
    )
    retrievers_by_document = _build_dense_retrievers(
        corpus,
        embeddings=cached_embeddings.embeddings,
        embedder=embedder,
        batch_size=batch_size,
    )
    evaluations = tuple(
        sorted(
            (
                _evaluate_dense_example(
                    example=example,
                    query_embedding=query_embeddings[index],
                    retrievers_by_document=retrievers_by_document,
                    k_values=selected_k_values,
                )
                for index, example in enumerate(examples)
            ),
            key=lambda evaluation: (
                evaluation.dataset.value,
                evaluation.example_id,
            ),
        )
    )
    metrics = aggregate_evaluations(evaluations)
    metrics_payload = _build_dense_metrics_payload(
        corpus=corpus,
        evaluations=evaluations,
        metrics=metrics,
        embedder=embedder,
        batch_size=batch_size,
        split=split,
        k_values=selected_k_values,
    )
    comparison_payload = _build_comparison_payload(
        bm25_metrics_path=bm25_metrics_path.resolve(),
        dense_metrics_payload=metrics_payload,
    )
    predictions_path = output_root / DENSE_PREDICTIONS_FILENAME
    metrics_path = output_root / DENSE_METRICS_FILENAME
    comparison_path = output_root / COMPARISON_FILENAME

    write_bytes_atomically(
        predictions_path,
        serialize_evaluations_jsonl(evaluations),
    )
    write_bytes_atomically(metrics_path, _serialize_json(metrics_payload))
    write_bytes_atomically(comparison_path, _serialize_json(comparison_payload))

    return DenseBenchmarkResult(
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


def _build_dense_retrievers(
    corpus: FrozenBenchmarkCorpus,
    *,
    embeddings: FloatMatrix,
    embedder: DenseEmbedder,
    batch_size: int,
) -> dict[str, DenseRetriever]:
    retrievers: dict[str, DenseRetriever] = {}
    row_start = 0

    for dataset in corpus.datasets:
        for document_id, chunks in dataset.chunks_by_document:
            row_end = row_start + len(chunks)
            retrievers[document_id] = DenseRetriever(
                chunks,
                embedder=embedder,
                batch_size=batch_size,
                document_embeddings=embeddings[row_start:row_end],
            )
            row_start = row_end

    if row_start != corpus.indexed_chunk_count:
        raise ValueError("Embedding rows do not align with the frozen chunk corpus.")

    return retrievers


def _evaluate_dense_example(
    *,
    example: DatasetExample,
    query_embedding: FloatVector,
    retrievers_by_document: Mapping[str, DenseRetriever],
    k_values: tuple[int, ...],
) -> QueryRetrievalEvaluation:
    try:
        retriever = retrievers_by_document[example.question.document_id]
    except KeyError as error:
        raise RetrievalEvaluationError(
            f"No dense index exists for example {example.example_id!r}."
        ) from error

    return evaluate_query(
        dataset=example.dataset,
        example_id=example.example_id,
        question=example.question.text,
        gold_source_element_ids=(fact.element_id for fact in example.supporting_facts),
        results=retriever.search_vector(
            query_embedding,
            top_k=max(k_values),
        ),
        k_values=k_values,
    )


def _build_dense_metrics_payload(
    *,
    corpus: FrozenBenchmarkCorpus,
    evaluations: tuple[QueryRetrievalEvaluation, ...],
    metrics: RetrievalMetrics,
    embedder: DenseEmbedder,
    batch_size: int,
    split: DatasetSplit,
    k_values: tuple[int, ...],
) -> dict[str, object]:
    return {
        "benchmark_config": {
            "batch_size": batch_size,
            "device": embedder.device,
            "document_encoding_strategy": DOCUMENT_ENCODING_STRATEGY,
            "embedding_dimension": embedder.dimension,
            "embedding_model": embedder.model_id,
            "embedding_model_revision": embedder.model_revision,
            "k_values": list(k_values),
            "normalization": "l2_unit",
            "query_encoding_strategy": QUERY_ENCODING_STRATEGY,
            "query_prompt": BGE_QUERY_PROMPT,
            "retrieval_scope": "question_document",
            "retriever_type": "dense_exact",
            "similarity_metric": "cosine_via_dot_product",
            "split": split.value,
            "stored_embedding_dtype": "float32",
            "tie_breaker": "chunk_id_ascending",
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
    dense_metrics_payload: dict[str, object],
) -> dict[str, object]:
    bm25_payload = _load_json_object(
        bm25_metrics_path,
        label="BM25 metrics",
    )
    _validate_comparable_benchmarks(
        bm25_payload=bm25_payload,
        dense_payload=dense_metrics_payload,
    )
    bm25_by_scope = _metrics_by_scope(bm25_payload)
    dense_by_scope = _metrics_by_scope(dense_metrics_payload)

    return {
        "benchmark_identity": dense_metrics_payload["benchmark_identity"],
        "bm25_metrics_sha256": sha256(bm25_metrics_path.read_bytes()).hexdigest(),
        "delta_definition": "dense_minus_bm25",
        "results": {
            scope: {
                "absolute_delta": _metric_delta(
                    bm25=cast(dict[str, object], bm25_by_scope[scope]),
                    dense=cast(dict[str, object], dense_by_scope[scope]),
                ),
                "bm25": bm25_by_scope[scope],
                "dense": dense_by_scope[scope],
            }
            for scope in sorted(dense_by_scope)
        },
    }


def _validate_comparable_benchmarks(
    *,
    bm25_payload: dict[str, object],
    dense_payload: dict[str, object],
) -> None:
    matching_fields = (
        "dataset_counts",
        "document_count",
        "indexed_chunk_count",
        "query_count",
    )

    for field in matching_fields:
        if bm25_payload.get(field) != dense_payload.get(field):
            raise ValueError(f"BM25 and dense benchmark field {field!r} must match.")

    bm25_config = _require_mapping(
        bm25_payload.get("benchmark_config"),
        label="BM25 benchmark_config",
    )
    dense_config = _require_mapping(
        dense_payload.get("benchmark_config"),
        label="Dense benchmark_config",
    )

    for field in ("k_values", "retrieval_scope", "split"):
        if bm25_config.get(field) != dense_config.get(field):
            raise ValueError(f"BM25 and dense benchmark config field {field!r} must match.")


def _metrics_by_scope(payload: dict[str, object]) -> dict[str, object]:
    combined = _require_mapping(payload.get("metrics"), label="combined metrics")
    datasets = _require_mapping(
        payload.get("metrics_by_dataset"),
        label="metrics_by_dataset",
    )
    return {"combined": combined, **datasets}


def _metric_delta(
    *,
    bm25: dict[str, object],
    dense: dict[str, object],
) -> dict[str, object]:
    bm25_hit = _require_mapping(bm25.get("hit_rate_at_k"), label="BM25 hit rates")
    dense_hit = _require_mapping(dense.get("hit_rate_at_k"), label="dense hit rates")
    bm25_recall = _require_mapping(bm25.get("recall_at_k"), label="BM25 recall")
    dense_recall = _require_mapping(dense.get("recall_at_k"), label="dense recall")

    return {
        "hit_rate_at_k": {
            key: _require_number(dense_hit.get(key), label=f"dense Hit@{key}")
            - _require_number(bm25_hit.get(key), label=f"BM25 Hit@{key}")
            for key in sorted(dense_hit, key=int)
        },
        "mrr": _require_number(dense.get("mrr"), label="dense MRR")
        - _require_number(bm25.get("mrr"), label="BM25 MRR"),
        "recall_at_k": {
            key: _require_number(dense_recall.get(key), label=f"dense Recall@{key}")
            - _require_number(bm25_recall.get(key), label=f"BM25 Recall@{key}")
            for key in sorted(dense_recall, key=int)
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
