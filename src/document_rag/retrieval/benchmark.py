"""Run and serialize BM25 benchmarks over normalized dataset artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from document_rag.datasets.models import (
    DatasetExample,
    DatasetName,
    DatasetSplit,
)
from document_rag.domain.documents import Document, DocumentElement
from document_rag.ingestion.chunking import DocumentChunk, RegexTokenCounter
from document_rag.retrieval.bm25 import (
    TOKENIZATION_STRATEGY,
    BM25Retriever,
    lexical_tokenize,
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

PREDICTIONS_FILENAME = "retrieval_predictions.jsonl"
METRICS_FILENAME = "retrieval_metrics.json"


@dataclass(frozen=True, slots=True)
class BenchmarkDatasetCounts:
    """Input counts for one normalized dataset."""

    document_count: int
    indexed_chunk_count: int
    query_count: int

    def to_record(self) -> dict[str, int]:
        """Convert counts to a JSON-compatible record."""

        return {
            "document_count": self.document_count,
            "indexed_chunk_count": self.indexed_chunk_count,
            "query_count": self.query_count,
        }


@dataclass(frozen=True, slots=True)
class BM25BenchmarkResult:
    """Artifacts and summary returned by a complete BM25 benchmark."""

    predictions_path: Path
    metrics_path: Path
    evaluations: tuple[QueryRetrievalEvaluation, ...]
    metrics: RetrievalMetrics
    dataset_counts: tuple[tuple[DatasetName, BenchmarkDatasetCounts], ...]
    indexed_chunk_count: int
    document_count: int


@dataclass(frozen=True, slots=True)
class _LoadedDataset:
    dataset: DatasetName
    documents: tuple[Document, ...]
    elements: tuple[DocumentElement, ...]
    examples: tuple[DatasetExample, ...]


def run_bm25_benchmark(
    *,
    dataset_directories: Mapping[DatasetName, Path],
    output_directory: Path,
    split: DatasetSplit = DatasetSplit.TEST,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
) -> BM25BenchmarkResult:
    """Evaluate deterministic per-document BM25 indexes and write artifacts."""

    if not dataset_directories:
        raise ValueError("At least one normalized dataset directory is required.")

    selected_k_values = normalize_k_values(k_values)
    loaded_datasets = tuple(
        _load_dataset(
            dataset=dataset,
            directory=Path(directory),
            split=split,
        )
        for dataset, directory in sorted(
            dataset_directories.items(),
            key=lambda item: item[0].value,
        )
    )

    all_evaluations: list[QueryRetrievalEvaluation] = []
    dataset_counts: list[tuple[DatasetName, BenchmarkDatasetCounts]] = []
    all_document_ids: set[str] = set()
    indexed_chunk_count = 0

    for loaded in loaded_datasets:
        duplicate_document_ids = all_document_ids & {
            document.document_id for document in loaded.documents
        }

        if duplicate_document_ids:
            duplicate = min(duplicate_document_ids)
            raise ValueError(f"Document ID appears in multiple datasets: {duplicate!r}.")

        all_document_ids.update(document.document_id for document in loaded.documents)
        chunks_by_document = _build_chunks_by_document(loaded)
        retrievers_by_document = {
            document_id: BM25Retriever(chunks) for document_id, chunks in chunks_by_document.items()
        }
        dataset_evaluations = tuple(
            _evaluate_example(
                example,
                retrievers_by_document=retrievers_by_document,
                k_values=selected_k_values,
            )
            for example in loaded.examples
        )
        dataset_chunk_count = sum(len(chunks) for chunks in chunks_by_document.values())

        all_evaluations.extend(dataset_evaluations)
        indexed_chunk_count += dataset_chunk_count
        dataset_counts.append(
            (
                loaded.dataset,
                BenchmarkDatasetCounts(
                    document_count=len(loaded.documents),
                    indexed_chunk_count=dataset_chunk_count,
                    query_count=len(dataset_evaluations),
                ),
            )
        )

    evaluations = tuple(
        sorted(
            all_evaluations,
            key=lambda evaluation: (
                evaluation.dataset.value,
                evaluation.example_id,
            ),
        )
    )
    metrics = aggregate_evaluations(evaluations)
    output_root = output_directory.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = output_root / PREDICTIONS_FILENAME
    metrics_path = output_root / METRICS_FILENAME

    prediction_bytes = b"".join(
        (
            json.dumps(
                evaluation.to_record(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for evaluation in evaluations
    )
    metrics_payload = _build_metrics_payload(
        evaluations=evaluations,
        metrics=metrics,
        dataset_counts=tuple(dataset_counts),
        indexed_chunk_count=indexed_chunk_count,
        document_count=len(all_document_ids),
        split=split,
        k_values=selected_k_values,
    )
    metrics_bytes = (
        json.dumps(
            metrics_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    _write_bytes_atomically(predictions_path, prediction_bytes)
    _write_bytes_atomically(metrics_path, metrics_bytes)

    return BM25BenchmarkResult(
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        evaluations=evaluations,
        metrics=metrics,
        dataset_counts=tuple(dataset_counts),
        indexed_chunk_count=indexed_chunk_count,
        document_count=len(all_document_ids),
    )


def _load_dataset(
    *,
    dataset: DatasetName,
    directory: Path,
    split: DatasetSplit,
) -> _LoadedDataset:
    root = directory.resolve()
    manifest_path = root / "manifest.json"

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(manifest_path) from None
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Invalid dataset manifest: {manifest_path}.") from error

    if not isinstance(manifest, dict) or manifest.get("dataset") != dataset.value:
        raise ValueError(f"Dataset manifest must identify {dataset.value!r}: {manifest_path}.")

    split_directory = root / split.value
    documents = _load_jsonl_models(
        split_directory / "documents.jsonl",
        Document,
    )
    elements = _load_jsonl_models(
        split_directory / "elements.jsonl",
        DocumentElement,
    )
    examples = _load_jsonl_models(
        split_directory / "examples.jsonl",
        DatasetExample,
    )
    loaded = _LoadedDataset(
        dataset=dataset,
        documents=documents,
        elements=elements,
        examples=examples,
    )
    _validate_loaded_dataset(loaded, split=split)
    return loaded


def _load_jsonl_models[ModelT: BaseModel](
    path: Path,
    model_type: type[ModelT],
) -> tuple[ModelT, ...]:
    if not path.is_file():
        raise FileNotFoundError(path)

    records: list[ModelT] = []

    with path.open("rb") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                records.append(model_type.model_validate_json(line))
            except ValidationError as error:
                raise ValueError(f"{path} line {line_number}: invalid record.") from error

    if not records:
        raise ValueError(f"Normalized artifact is empty: {path}.")

    return tuple(records)


def _validate_loaded_dataset(
    loaded: _LoadedDataset,
    *,
    split: DatasetSplit,
) -> None:
    documents_by_id = _unique_by_id(
        loaded.documents,
        id_getter=lambda document: document.document_id,
        label="document",
    )
    elements_by_id = _unique_by_id(
        loaded.elements,
        id_getter=lambda element: element.element_id,
        label="element",
    )
    _unique_by_id(
        loaded.examples,
        id_getter=lambda example: example.example_id,
        label="example",
    )

    for element in loaded.elements:
        if element.document_id not in documents_by_id:
            raise ValueError(f"Element references unknown document: {element.element_id!r}.")

        if (
            element.parent_element_id is not None
            and element.parent_element_id not in elements_by_id
        ):
            raise ValueError(f"Element references unknown parent: {element.element_id!r}.")

    for example in loaded.examples:
        if example.dataset is not loaded.dataset:
            raise ValueError(f"Example {example.example_id!r} has the wrong dataset.")

        if example.split is not split:
            raise ValueError(f"Example {example.example_id!r} has the wrong split.")

        if example.question.document_id not in documents_by_id:
            raise ValueError(f"Example references unknown document: {example.example_id!r}.")

        for fact in example.supporting_facts:
            if fact.element_id not in elements_by_id:
                raise ValueError(
                    f"Example {example.example_id!r} references unknown gold "
                    f"element {fact.element_id!r}."
                )


def _unique_by_id[ItemT](
    items: Iterable[ItemT],
    *,
    id_getter: Any,
    label: str,
) -> dict[str, ItemT]:
    indexed: dict[str, ItemT] = {}

    for item in items:
        item_id = cast(str, id_getter(item))

        if item_id in indexed:
            raise ValueError(f"Duplicate {label} ID: {item_id!r}.")

        indexed[item_id] = item

    return indexed


def _build_chunks_by_document(
    loaded: _LoadedDataset,
) -> dict[str, tuple[DocumentChunk, ...]]:
    documents_by_id = {document.document_id: document for document in loaded.documents}
    elements_by_document: dict[str, list[DocumentElement]] = defaultdict(list)

    for element in loaded.elements:
        elements_by_document[element.document_id].append(element)

    chunks_by_document: dict[str, tuple[DocumentChunk, ...]] = {}

    for document_id in sorted(documents_by_id):
        elements = tuple(elements_by_document.get(document_id, ()))

        if not elements:
            raise ValueError(f"Document has no retrieval elements: {document_id!r}.")

        chunks_by_document[document_id] = _build_document_chunks(
            documents_by_id[document_id],
            elements,
        )

    return chunks_by_document


def _build_document_chunks(
    document: Document,
    elements: tuple[DocumentElement, ...],
) -> tuple[DocumentChunk, ...]:
    children_by_parent: dict[str, list[DocumentElement]] = defaultdict(list)

    for element in elements:
        if element.parent_element_id is not None:
            children_by_parent[element.parent_element_id].append(element)

    top_level_elements = tuple(
        sorted(
            (
                element
                for element in elements
                if element.parent_element_id is None and lexical_tokenize(element.source_text)
            ),
            key=lambda element: (element.page_number, element.element_id),
        )
    )

    if not top_level_elements:
        raise ValueError(
            f"Document has no lexically indexable top-level elements: {document.document_id!r}."
        )

    document_sha256 = document.checksum_sha256 or _hash_elements(elements)
    token_counter = RegexTokenCounter()
    chunks: list[DocumentChunk] = []

    for chunk_index, element in enumerate(top_level_elements):
        child_ids = tuple(
            child.element_id
            for child in sorted(
                children_by_parent.get(element.element_id, ()),
                key=lambda child: (
                    child.table_coordinates.row_index
                    if child.table_coordinates is not None
                    and child.table_coordinates.row_index is not None
                    else -1,
                    child.element_id,
                ),
            )
        )
        source_element_ids = (element.element_id, *child_ids)
        chunk_id = _build_normalized_chunk_id(
            document_id=document.document_id,
            source_element_id=element.element_id,
            text=element.source_text,
        )

        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                document_sha256=document_sha256,
                filename=document.file_name,
                chunk_index=chunk_index,
                page_start=element.page_number,
                page_end=element.page_number,
                source_element_ids=source_element_ids,
                text=element.source_text,
                char_count=len(element.source_text),
                token_count=token_counter.count(element.source_text),
                block_count=1,
            )
        )

    return tuple(chunks)


def _evaluate_example(
    example: DatasetExample,
    *,
    retrievers_by_document: Mapping[str, BM25Retriever],
    k_values: tuple[int, ...],
) -> QueryRetrievalEvaluation:
    try:
        retriever = retrievers_by_document[example.question.document_id]
    except KeyError as error:
        raise RetrievalEvaluationError(
            f"No BM25 index exists for example {example.example_id!r}."
        ) from error

    results = retriever.search(
        example.question.text,
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


def _build_metrics_payload(
    *,
    evaluations: tuple[QueryRetrievalEvaluation, ...],
    metrics: RetrievalMetrics,
    dataset_counts: tuple[tuple[DatasetName, BenchmarkDatasetCounts], ...],
    indexed_chunk_count: int,
    document_count: int,
    split: DatasetSplit,
    k_values: tuple[int, ...],
) -> dict[str, object]:
    metrics_by_dataset = {
        dataset.value: aggregate_evaluations(
            evaluation for evaluation in evaluations if evaluation.dataset is dataset
        ).to_record()
        for dataset, _ in dataset_counts
    }

    return {
        "benchmark_config": {
            "k_values": list(k_values),
            "retrieval_scope": "question_document",
            "retriever_type": "bm25_okapi",
            "split": split.value,
            "tokenization_strategy": TOKENIZATION_STRATEGY,
        },
        "dataset_counts": {dataset.value: counts.to_record() for dataset, counts in dataset_counts},
        "document_count": document_count,
        "indexed_chunk_count": indexed_chunk_count,
        "metrics": metrics.to_record(),
        "metrics_by_dataset": metrics_by_dataset,
        "query_count": len(evaluations),
    }


def _hash_elements(elements: Iterable[DocumentElement]) -> str:
    digest = sha256()

    for element in sorted(elements, key=lambda item: item.element_id):
        _update_length_prefixed(digest, element.element_id)
        _update_length_prefixed(digest, element.source_text)

    return digest.hexdigest()


def _build_normalized_chunk_id(
    *,
    document_id: str,
    source_element_id: str,
    text: str,
) -> str:
    digest = sha256()

    for value in (document_id, source_element_id, text):
        _update_length_prefixed(digest, value)

    return f"normalized-chunk:{digest.hexdigest()}"


def _update_length_prefixed(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(length=8, byteorder="big"))
    digest.update(encoded)


def _write_bytes_atomically(path: Path, content: bytes) -> None:
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        temporary_file.write(content)
        temporary_file.flush()

    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
