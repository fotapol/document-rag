"""Prepare deterministic, production-shaped RAG chat data for adapter training."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from pydantic import BaseModel, ValidationError

from document_rag.datasets.models import DatasetExample, DatasetName, DatasetSplit
from document_rag.domain.documents import Document, DocumentElement, DocumentElementType
from document_rag.ingestion.chunking import DocumentChunk, RegexTokenCounter
from document_rag.rag.config import RAGConfig
from document_rag.rag.prompting import SYSTEM_INSTRUCTION, UNSUPPORTED_ANSWER, build_grounded_prompt
from document_rag.rag.service import InMemoryHybridIndexFactory, RetrieverFactory
from document_rag.retrieval.benchmark import write_bytes_atomically
from document_rag.retrieval.models import RetrievalResult

RAG_TRAINING_SCHEMA_VERSION = 2
DEFAULT_REFUSAL_RATIO = 0.2
type RAGTrainingCategory = Literal[
    "simple_lookup",
    "table_lookup",
    "reasoning",
    "unsupported",
]
type RAGTrainingContextMode = Literal[
    "retrieved",
    "oracle_augmented",
    "retrieval_insufficient",
]
type UnitStatus = Literal["preserved", "inferred", "not_applicable"]

_NUMBER_PATTERN = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?")
_NUMERIC_ANSWER_PATTERN = re.compile(r"^[-+]?\d[\d,]*(?:\.\d+)?$")
_PERCENT_QUESTION_PATTERN = re.compile(r"\b(?:percent|percentage)\b", re.IGNORECASE)
_BASIS_POINT_PATTERN = re.compile(r"\bbasis points?\b", re.IGNORECASE)
_SCALE_PATTERN = re.compile(r"\b(thousand|million|billion)\b", re.IGNORECASE)
_CURRENCY_WORD_PATTERN = re.compile(r"\b(?:dollars?|usd)\b", re.IGNORECASE)
_EXPLICIT_UNIT_PATTERN = re.compile(
    r"(?:[$€£]|%|\b(?:percent|percentage|basis points?|thousand|million|billion)\b)",
    re.IGNORECASE,
)
_ADDITIVE_PROGRAM_PATTERN = re.compile(r"^\s*(?:add|subtract)\s*\(", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class RAGTrainingExportConfig:
    """Policies that shape supported and unsupported training examples."""

    refusal_ratio: float = DEFAULT_REFUSAL_RATIO
    oracle_augment: bool = True
    document_resplit: bool = True

    def __post_init__(self) -> None:
        """Reject ratios that would overwhelm supported-answer supervision."""

        if not 0.0 <= self.refusal_ratio < 0.5:
            raise ValueError("refusal_ratio must be greater than or equal to 0 and below 0.5.")


@dataclass(frozen=True, slots=True)
class RAGTrainingArtifact:
    """Metadata for one exported RAG-aligned split."""

    split: DatasetSplit
    path: Path
    record_count: int
    supported_count: int
    refusal_count: int
    oracle_augmented_count: int
    ambiguous_unit_exclusion_count: int
    gold_source_overflow_exclusion_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class RAGTrainingExportResult:
    """Complete RAG-aligned export and its reproducibility manifest."""

    artifacts: tuple[RAGTrainingArtifact, ...]
    manifest_path: Path
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class _LoadedSplit:
    dataset: DatasetName
    split: DatasetSplit
    documents: tuple[Document, ...]
    elements: tuple[DocumentElement, ...]
    examples: tuple[DatasetExample, ...]


@dataclass(frozen=True, slots=True)
class _GroundedAnswer:
    text: str
    expected_units: tuple[str, ...]
    unit_status: UnitStatus


def export_rag_training_data(
    *,
    finqa_directory: Path,
    docfinqa_directory: Path,
    output_directory: Path,
    splits: Sequence[DatasetSplit] = tuple(DatasetSplit),
    rag_config: RAGConfig | None = None,
    export_config: RAGTrainingExportConfig | None = None,
    retriever_factory: RetrieverFactory | None = None,
) -> RAGTrainingExportResult:
    """Export source-grounded chat records using the production prompt and retrieval shape."""

    selected_splits = _validate_splits(splits)
    resolved_rag_config = rag_config or RAGConfig()
    resolved_export_config = export_config or RAGTrainingExportConfig()
    resolved_retriever_factory = retriever_factory or InMemoryHybridIndexFactory(
        resolved_rag_config
    )
    dataset_roots = {
        DatasetName.FINQA: finqa_directory.resolve(),
        DatasetName.DOCFINQA: docfinqa_directory.resolve(),
    }
    input_manifests = {
        dataset: _load_and_validate_manifest(root, dataset=dataset)
        for dataset, root in dataset_roots.items()
    }
    source_splits = (
        tuple(DatasetSplit) if resolved_export_config.document_resplit else selected_splits
    )
    loaded_splits = tuple(
        _load_split(dataset=dataset, root=dataset_roots[dataset], split=split)
        for split in source_splits
        for dataset in sorted(dataset_roots, key=lambda item: item.value)
    )

    if not resolved_export_config.document_resplit:
        _validate_document_group_isolation(loaded_splits)

    output_root = output_directory.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts: list[RAGTrainingArtifact] = []
    artifact_distributions: dict[DatasetSplit, dict[str, object]] = {}
    all_records: list[dict[str, object]] = []
    ambiguous_exclusion_groups: list[str] = []
    gold_overflow_exclusion_groups: list[str] = []

    for loaded in loaded_splits:
        dataset_records, dataset_ambiguous_groups, dataset_overflow_groups = _build_split_records(
            loaded,
            rag_config=resolved_rag_config,
            export_config=resolved_export_config,
            retriever_factory=resolved_retriever_factory,
        )
        all_records.extend(dataset_records)
        ambiguous_exclusion_groups.extend(dataset_ambiguous_groups)
        gold_overflow_exclusion_groups.extend(dataset_overflow_groups)

    if resolved_export_config.document_resplit:
        all_records = [
            {
                **record,
                "split": _assigned_document_split(cast(str, record["document_group_id"])).value,
            }
            for record in all_records
        ]

    for split in selected_splits:
        records = [record for record in all_records if record["split"] == split.value]
        supported_count = sum(cast(bool, record["answerable"]) for record in records)
        refusal_count = len(records) - supported_count
        oracle_augmented_count = sum(
            record["context_mode"] == "oracle_augmented" for record in records
        )
        ambiguous_unit_exclusion_count = sum(
            (
                _assigned_document_split(group_id) is split
                if resolved_export_config.document_resplit
                else group_id.startswith(f"{split.value}:")
            )
            for group_id in ambiguous_exclusion_groups
        )
        gold_source_overflow_exclusion_count = sum(
            (
                _assigned_document_split(group_id) is split
                if resolved_export_config.document_resplit
                else group_id.startswith(f"{split.value}:")
            )
            for group_id in gold_overflow_exclusion_groups
        )

        records.sort(
            key=lambda record: (
                cast(str, record["dataset"]),
                cast(str, record["example_id"]),
            )
        )
        _validate_records(records, split=split)
        artifact_distributions[split] = {
            "category_counts": _count_record_field(records, "category"),
            "context_mode_counts": _count_record_field(records, "context_mode"),
            "dataset_counts": _count_record_field(records, "dataset"),
            "document_group_count": len(
                {cast(str, record["document_group_id"]) for record in records}
            ),
            "unit_status_counts": _count_record_field(records, "unit_status"),
        }
        content = _serialize_jsonl(records)
        path = output_root / f"{split.value}.jsonl"
        write_bytes_atomically(path, content)
        artifacts.append(
            RAGTrainingArtifact(
                split=split,
                path=path,
                record_count=len(records),
                supported_count=supported_count,
                refusal_count=refusal_count,
                oracle_augmented_count=oracle_augmented_count,
                ambiguous_unit_exclusion_count=ambiguous_unit_exclusion_count,
                gold_source_overflow_exclusion_count=(gold_source_overflow_exclusion_count),
                sha256=sha256(content).hexdigest(),
            )
        )

    manifest_payload = {
        "artifacts": {
            artifact.split.value: {
                "ambiguous_unit_exclusion_count": artifact.ambiguous_unit_exclusion_count,
                "oracle_augmented_count": artifact.oracle_augmented_count,
                "gold_source_overflow_exclusion_count": (
                    artifact.gold_source_overflow_exclusion_count
                ),
                "path": artifact.path.relative_to(output_root).as_posix(),
                "record_count": artifact.record_count,
                "refusal_count": artifact.refusal_count,
                "sha256": artifact.sha256,
                "supported_count": artifact.supported_count,
                **artifact_distributions[artifact.split],
            }
            for artifact in artifacts
        },
        "format": "rag_chat_jsonl",
        "inputs": {
            dataset.value: {
                "manifest_sha256": _sha256_file(dataset_roots[dataset] / "manifest.json"),
                "source": input_manifests[dataset].get("source"),
                "sources": input_manifests[dataset].get("sources"),
            }
            for dataset in sorted(dataset_roots, key=lambda item: item.value)
        },
        "policies": {
            "ambiguous_units": "exclude_supported_example",
            "answerable_context": "all_gold_source_lineage_required",
            "document_split": "sha256_source-report-group_80_10_10",
            "document_resplit": resolved_export_config.document_resplit,
            "oracle_augment": resolved_export_config.oracle_augment,
            "refusal_ratio": resolved_export_config.refusal_ratio,
            "unsupported_target": UNSUPPORTED_ANSWER,
        },
        "retrieval": {
            "candidate_k": resolved_rag_config.candidate_k,
            "embedding_model_id": resolved_rag_config.embedding_model_id,
            "embedding_model_revision": resolved_rag_config.embedding_model_revision,
            "max_table_rows_per_parent": resolved_rag_config.max_table_rows_per_parent,
            "rrf_k": resolved_rag_config.rrf_k,
            "top_k": resolved_rag_config.top_k,
        },
        "schema_version": RAG_TRAINING_SCHEMA_VERSION,
        "system_instruction": SYSTEM_INSTRUCTION,
    }
    manifest_content = (
        json.dumps(
            manifest_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    manifest_path = output_root / "manifest.json"
    write_bytes_atomically(manifest_path, manifest_content)
    return RAGTrainingExportResult(
        artifacts=tuple(artifacts),
        manifest_path=manifest_path,
        manifest_sha256=sha256(manifest_content).hexdigest(),
    )


def _build_split_records(
    loaded: _LoadedSplit,
    *,
    rag_config: RAGConfig,
    export_config: RAGTrainingExportConfig,
    retriever_factory: RetrieverFactory,
) -> tuple[list[dict[str, object]], tuple[str, ...], tuple[str, ...]]:
    documents_by_id = {document.document_id: document for document in loaded.documents}
    elements_by_document: dict[str, list[DocumentElement]] = defaultdict(list)
    examples_by_document: dict[str, list[DatasetExample]] = defaultdict(list)

    for element in loaded.elements:
        elements_by_document[element.document_id].append(element)

    for example in loaded.examples:
        examples_by_document[example.question.document_id].append(example)

    records: list[dict[str, object]] = []
    negative_candidates: list[dict[str, object]] = []
    ambiguous_exclusion_groups: list[str] = []
    gold_overflow_exclusion_groups: list[str] = []
    supported_count = 0

    for document_id in sorted(examples_by_document):
        document = documents_by_id[document_id]
        chunks = _build_training_chunks(document, elements_by_document[document_id])
        chunks_by_source = {
            source_id: chunk for chunk in chunks for source_id in chunk.source_element_ids
        }
        retriever = retriever_factory.build(chunks)

        for example in sorted(examples_by_document[document_id], key=lambda item: item.example_id):
            deep_results = retriever.search(
                example.question.text,
                top_k=rag_config.candidate_k,
            )
            production_results = _rerank(deep_results[: rag_config.top_k])
            gold_ids = tuple(fact.element_id for fact in example.supporting_facts)
            context_mode: RAGTrainingContextMode = "retrieved"

            if len(set(gold_ids)) > rag_config.top_k:
                negative = _build_refusal_record(
                    loaded=loaded,
                    document=document,
                    example=example,
                    results=production_results,
                    mode="retrieval_insufficient",
                )

                if negative is not None:
                    negative_candidates.append(negative)

                group_id = _document_group_id(document, example)
                gold_overflow_exclusion_groups.append(
                    group_id
                    if export_config.document_resplit
                    else f"{loaded.split.value}:{group_id}"
                )
                continue

            if not _contains_all_gold(production_results, gold_ids):
                negative = _build_refusal_record(
                    loaded=loaded,
                    document=document,
                    example=example,
                    results=production_results,
                    mode="retrieval_insufficient",
                )

                if negative is not None:
                    negative_candidates.append(negative)

                if not export_config.oracle_augment:
                    continue

                supported_results = _oracle_augment(
                    production_results,
                    gold_ids=gold_ids,
                    chunks_by_source=chunks_by_source,
                    top_k=rag_config.top_k,
                )
                context_mode = "oracle_augmented"
            else:
                supported_results = production_results
                negative_results = _without_gold(
                    deep_results,
                    gold_ids=gold_ids,
                    top_k=rag_config.top_k,
                )
                negative = _build_refusal_record(
                    loaded=loaded,
                    document=document,
                    example=example,
                    results=negative_results,
                    mode="retrieval_insufficient",
                )

                if negative is not None:
                    negative_candidates.append(negative)

            grounded_answer = _build_grounded_answer(
                example,
                results=supported_results,
                gold_ids=gold_ids,
            )

            if grounded_answer is None:
                group_id = _document_group_id(document, example)
                ambiguous_exclusion_groups.append(
                    group_id
                    if export_config.document_resplit
                    else f"{loaded.split.value}:{group_id}"
                )
                continue

            records.append(
                _build_supported_record(
                    loaded=loaded,
                    document=document,
                    example=example,
                    results=supported_results,
                    context_mode=context_mode,
                    grounded_answer=grounded_answer,
                    gold_ids=gold_ids,
                )
            )
            supported_count += 1

    refusal_target = _refusal_target(
        supported_count=supported_count,
        refusal_ratio=export_config.refusal_ratio,
    )
    selected_negatives = tuple(
        sorted(
            negative_candidates,
            key=lambda record: (
                _stable_digest(cast(str, record["example_id"])),
                cast(str, record["example_id"]),
            ),
        )[:refusal_target]
    )
    records.extend(selected_negatives)
    return (
        records,
        tuple(ambiguous_exclusion_groups),
        tuple(gold_overflow_exclusion_groups),
    )


def _build_supported_record(
    *,
    loaded: _LoadedSplit,
    document: Document,
    example: DatasetExample,
    results: tuple[RetrievalResult, ...],
    context_mode: RAGTrainingContextMode,
    grounded_answer: _GroundedAnswer,
    gold_ids: tuple[str, ...],
) -> dict[str, object]:
    source_numbers = _gold_source_numbers(results, gold_ids)
    prompt = build_grounded_prompt(question=example.question.text, results=results)
    messages = [*prompt.to_messages(), {"role": "assistant", "content": grounded_answer.text}]
    return {
        "answerable": True,
        "category": _category(example, results=results, gold_ids=gold_ids),
        "context_mode": context_mode,
        "dataset": loaded.dataset.value,
        "document_group_id": _document_group_id(document, example),
        "document_id": document.document_id,
        "example_id": f"{example.example_id}:supported",
        "expected_units": list(grounded_answer.expected_units),
        "expected_values": [example.reference_answer.text],
        "gold_source_element_ids": list(gold_ids),
        "gold_source_numbers": list(source_numbers),
        "messages": messages,
        "reasoning_program": example.reference_answer.normalized_program
        or example.reference_answer.program,
        "source_chunk_ids": [result.chunk_id for result in results],
        "source_split": loaded.split.value,
        "split": loaded.split.value,
        "unit_status": grounded_answer.unit_status,
    }


def _build_refusal_record(
    *,
    loaded: _LoadedSplit,
    document: Document,
    example: DatasetExample,
    results: tuple[RetrievalResult, ...],
    mode: Literal["retrieval_insufficient"],
) -> dict[str, object] | None:
    if not results or _context_contains_answer(results, example.reference_answer.text):
        return None

    prompt = build_grounded_prompt(question=example.question.text, results=results)
    return {
        "answerable": False,
        "category": "unsupported",
        "context_mode": mode,
        "dataset": loaded.dataset.value,
        "document_group_id": _document_group_id(document, example),
        "document_id": document.document_id,
        "example_id": f"{example.example_id}:unsupported",
        "expected_units": [],
        "expected_values": [],
        "gold_source_element_ids": [],
        "gold_source_numbers": [],
        "messages": [
            *prompt.to_messages(),
            {"role": "assistant", "content": UNSUPPORTED_ANSWER},
        ],
        "reasoning_program": None,
        "source_chunk_ids": [result.chunk_id for result in results],
        "source_split": loaded.split.value,
        "split": loaded.split.value,
        "unit_status": "not_applicable",
    }


def _build_grounded_answer(
    example: DatasetExample,
    *,
    results: tuple[RetrievalResult, ...],
    gold_ids: tuple[str, ...],
) -> _GroundedAnswer | None:
    source_numbers = _gold_source_numbers(results, gold_ids)

    if not source_numbers:
        raise ValueError(f"Example {example.example_id!r} has no cited gold source.")

    gold_texts = tuple(
        result.chunk.text
        for result in results
        if set(result.chunk.source_element_ids) & set(gold_ids)
    )
    formatted = _format_answer_with_units(
        answer=example.reference_answer.text,
        question=example.question.text,
        evidence_texts=gold_texts,
        program=example.reference_answer.normalized_program or example.reference_answer.program,
    )

    if formatted is None:
        return None

    explanation = (example.reference_answer.explanation or "").strip()
    sentence = explanation.rstrip(".") if explanation else f"The answer is {formatted.text}"

    if formatted.text.casefold() not in sentence.casefold():
        sentence = f"{sentence}. The answer is {formatted.text}"

    citations = " ".join(f"[Source {number}]" for number in source_numbers)
    return _GroundedAnswer(
        text=f"{sentence}. {citations}",
        expected_units=formatted.expected_units,
        unit_status=formatted.unit_status,
    )


def _format_answer_with_units(
    *,
    answer: str,
    question: str,
    evidence_texts: tuple[str, ...],
    program: str | None,
) -> _GroundedAnswer | None:
    normalized_answer = answer.strip().rstrip(".")

    if _EXPLICIT_UNIT_PATTERN.search(normalized_answer):
        return _GroundedAnswer(
            text=normalized_answer,
            expected_units=_extract_unit_labels(normalized_answer),
            unit_status="preserved",
        )

    if _NUMERIC_ANSWER_PATTERN.fullmatch(normalized_answer) is None:
        return _GroundedAnswer(
            text=normalized_answer,
            expected_units=(),
            unit_status="not_applicable",
        )

    combined_evidence = "\n".join(evidence_texts)
    prefix = ""
    suffix_parts: list[str] = []

    if _BASIS_POINT_PATTERN.search(question):
        suffix_parts.append("basis points")
    elif _PERCENT_QUESTION_PATTERN.search(question):
        suffix_parts.append("%")

    scale_match = _SCALE_PATTERN.search(question)

    if scale_match is not None:
        suffix_parts.append(scale_match.group(1).casefold())

    if _CURRENCY_WORD_PATTERN.search(question):
        prefix = "$"

    exact_units = _units_for_exact_value(normalized_answer, combined_evidence)

    if exact_units is not None:
        exact_prefix, exact_suffixes = exact_units
        prefix = prefix or exact_prefix

        for suffix in exact_suffixes:
            if suffix not in suffix_parts:
                suffix_parts.append(suffix)

    if (
        not prefix
        and "$" in combined_evidence
        and program is not None
        and _ADDITIVE_PROGRAM_PATTERN.search(program)
    ):
        prefix = "$"

    if not prefix and not suffix_parts:
        if _EXPLICIT_UNIT_PATTERN.search(combined_evidence):
            return None

        return _GroundedAnswer(
            text=normalized_answer,
            expected_units=(),
            unit_status="not_applicable",
        )

    suffix = "".join(f"{part}" if part == "%" else f" {part}" for part in suffix_parts)
    expected_units = tuple(unit for unit in (prefix, *suffix_parts) if unit)
    return _GroundedAnswer(
        text=f"{prefix}{normalized_answer}{suffix}",
        expected_units=expected_units,
        unit_status="inferred",
    )


def _units_for_exact_value(answer: str, evidence: str) -> tuple[str, tuple[str, ...]] | None:
    try:
        expected = Decimal(answer.replace(",", ""))
    except InvalidOperation:
        return None

    matches: set[tuple[str, tuple[str, ...]]] = set()

    for number_match in _NUMBER_PATTERN.finditer(evidence):
        try:
            actual = Decimal(number_match.group(0).replace(",", ""))
        except InvalidOperation:
            continue

        if actual != expected:
            continue

        prefix_text = evidence[max(0, number_match.start() - 3) : number_match.start()]
        suffix_text = evidence[number_match.end() : number_match.end() + 24]
        prefix_match = re.search(r"[$€£]\s*$", prefix_text)
        suffix_matches: list[str] = []

        if re.match(r"\s*%", suffix_text):
            suffix_matches.append("%")

        scale_match = re.match(
            r"\s*(thousand|million|billion|basis points?)\b",
            suffix_text,
            re.IGNORECASE,
        )

        if scale_match is not None:
            suffix_matches.append(scale_match.group(1).casefold())

        matches.add(
            (
                prefix_match.group(0).strip() if prefix_match is not None else "",
                tuple(suffix_matches),
            )
        )

    if len(matches) != 1:
        return None

    return next(iter(matches))


def _extract_unit_labels(text: str) -> tuple[str, ...]:
    labels: list[str] = []

    for symbol in ("$", "€", "£", "%"):
        if symbol in text:
            labels.append(symbol)

    for match in re.finditer(
        r"\b(?:percent|percentage|basis points?|thousand|million|billion)\b",
        text,
        re.IGNORECASE,
    ):
        label = match.group(0).casefold()

        if label not in labels:
            labels.append(label)

    return tuple(labels)


def _category(
    example: DatasetExample,
    *,
    results: tuple[RetrievalResult, ...],
    gold_ids: tuple[str, ...],
) -> RAGTrainingCategory:
    if example.reference_answer.program or example.reference_answer.steps:
        return "reasoning"

    if any(
        result.chunk.retrieval_unit_kind == "table_row"
        and set(result.chunk.source_element_ids) & set(gold_ids)
        for result in results
    ):
        return "table_lookup"

    return "simple_lookup"


def _oracle_augment(
    results: tuple[RetrievalResult, ...],
    *,
    gold_ids: tuple[str, ...],
    chunks_by_source: Mapping[str, DocumentChunk],
    top_k: int,
) -> tuple[RetrievalResult, ...]:
    required_chunks: dict[str, DocumentChunk] = {}

    for gold_id in gold_ids:
        try:
            chunk = chunks_by_source[gold_id]
        except KeyError as error:
            raise ValueError(f"Gold source has no retrieval chunk: {gold_id!r}.") from error

        required_chunks[chunk.chunk_id] = chunk

    if len(required_chunks) > top_k:
        raise ValueError("Gold evidence requires more chunks than configured top_k.")

    required_ids = set(required_chunks)
    selected = list(results)

    for chunk_id, chunk in sorted(required_chunks.items()):
        if any(result.chunk_id == chunk_id for result in selected):
            continue

        selected.append(RetrievalResult(chunk=chunk, score=0.0, rank=len(selected) + 1))

    while len(selected) > top_k:
        removable_index = next(
            (
                index
                for index in range(len(selected) - 1, -1, -1)
                if selected[index].chunk_id not in required_ids
            ),
            None,
        )

        if removable_index is None:
            raise ValueError("Could not fit required gold evidence inside top_k.")

        selected.pop(removable_index)

    augmented = _rerank(selected)

    if not _contains_all_gold(augmented, gold_ids):
        raise ValueError("Oracle augmentation did not preserve every gold source.")

    return augmented


def _without_gold(
    results: Iterable[RetrievalResult],
    *,
    gold_ids: tuple[str, ...],
    top_k: int,
) -> tuple[RetrievalResult, ...]:
    gold = set(gold_ids)
    return _rerank(
        tuple(result for result in results if not (set(result.chunk.source_element_ids) & gold))[
            :top_k
        ]
    )


def _contains_all_gold(results: Iterable[RetrievalResult], gold_ids: Iterable[str]) -> bool:
    retrieved_lineage = {
        source_id for result in results for source_id in result.chunk.source_element_ids
    }
    return set(gold_ids) <= retrieved_lineage


def _gold_source_numbers(
    results: tuple[RetrievalResult, ...],
    gold_ids: tuple[str, ...],
) -> tuple[int, ...]:
    gold = set(gold_ids)
    return tuple(result.rank for result in results if set(result.chunk.source_element_ids) & gold)


def _context_contains_answer(results: Iterable[RetrievalResult], answer: str) -> bool:
    normalized_answer = answer.strip().rstrip(".")
    context = "\n".join(result.chunk.text for result in results)

    if _NUMERIC_ANSWER_PATTERN.fullmatch(normalized_answer):
        try:
            expected = Decimal(normalized_answer.replace(",", ""))
        except InvalidOperation:
            return True

        for match in _NUMBER_PATTERN.finditer(context):
            try:
                if Decimal(match.group(0).replace(",", "")) == expected:
                    return True
            except InvalidOperation:
                continue

        return False

    return normalized_answer.casefold() in context.casefold()


def _rerank(results: Iterable[RetrievalResult]) -> tuple[RetrievalResult, ...]:
    return tuple(replace(result, rank=rank) for rank, result in enumerate(results, start=1))


def _build_training_chunks(
    document: Document,
    elements: Iterable[DocumentElement],
) -> tuple[DocumentChunk, ...]:
    materialized = tuple(elements)
    table_ids_with_rows = {
        element.parent_element_id
        for element in materialized
        if element.element_type is DocumentElementType.TABLE_ROW
        and element.parent_element_id is not None
    }
    document_sha256 = document.checksum_sha256 or _hash_elements(materialized)
    token_counter = RegexTokenCounter()
    chunks: list[DocumentChunk] = []

    for chunk_index, element in enumerate(
        sorted(materialized, key=lambda item: (item.page_number, item.element_id))
    ):
        if (
            element.element_type is DocumentElementType.TABLE
            and element.element_id in table_ids_with_rows
        ):
            continue

        text = _render_element(element)
        chunk_id = (
            f"training-chunk:{_stable_digest(document.document_id, element.element_id, text)}"
        )
        is_row = element.element_type is DocumentElementType.TABLE_ROW
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                document_sha256=document_sha256,
                filename=document.file_name,
                chunk_index=chunk_index,
                page_start=element.page_number,
                page_end=element.page_number,
                source_element_ids=(element.element_id,),
                text=text,
                char_count=len(text),
                token_count=token_counter.count(text),
                block_count=1,
                retrieval_unit_kind="table_row" if is_row else "chunk",
                parent_chunk_id=(
                    f"training-parent:{_stable_digest(document.document_id, element.parent_element_id)}"
                    if is_row and element.parent_element_id is not None
                    else None
                ),
                parent_source_element_ids=(
                    (element.parent_element_id,)
                    if is_row and element.parent_element_id is not None
                    else ()
                ),
            )
        )

    if not chunks:
        raise ValueError(f"Document has no training retrieval chunks: {document.document_id!r}.")

    return tuple(chunks)


def _render_element(element: DocumentElement) -> str:
    if element.element_type is not DocumentElementType.TABLE_ROW:
        return element.source_text

    raw_headers = element.metadata.get("column_headers")

    if not isinstance(raw_headers, list) or not all(
        isinstance(value, str) for value in raw_headers
    ):
        return f"Table row:\n{element.source_text}"

    headers = cast(list[str], raw_headers)
    cells = tuple(cell.strip() for cell in element.source_text.split("|"))
    fields = " | ".join(
        f"{headers[index] if index < len(headers) and headers[index] else f'Column {index + 1}'}: {cell}"
        for index, cell in enumerate(cells)
    )
    title = element.section or "Financial table"
    return f"Table: {title}\n{fields}"


def _load_split(*, dataset: DatasetName, root: Path, split: DatasetSplit) -> _LoadedSplit:
    split_root = root / split.value
    documents = _load_jsonl_models(split_root / "documents.jsonl", Document)
    elements = _load_jsonl_models(split_root / "elements.jsonl", DocumentElement)
    examples = _load_jsonl_models(split_root / "examples.jsonl", DatasetExample)
    documents_by_id = {document.document_id: document for document in documents}
    elements_by_id = {element.element_id: element for element in elements}

    if len(documents_by_id) != len(documents):
        raise ValueError(f"Duplicate document IDs in {split_root}.")

    if len(elements_by_id) != len(elements):
        raise ValueError(f"Duplicate element IDs in {split_root}.")

    for example in examples:
        if example.dataset is not dataset or example.split is not split:
            raise ValueError(f"Example has the wrong dataset or split: {example.example_id!r}.")

        if example.question.document_id not in documents_by_id:
            raise ValueError(f"Example references an unknown document: {example.example_id!r}.")

        for fact in example.supporting_facts:
            if fact.element_id not in elements_by_id:
                raise ValueError(f"Example references an unknown element: {example.example_id!r}.")

    return _LoadedSplit(
        dataset=dataset,
        split=split,
        documents=documents,
        elements=elements,
        examples=examples,
    )


def _load_jsonl_models[ModelT: BaseModel](path: Path, model: type[ModelT]) -> tuple[ModelT, ...]:
    if not path.is_file():
        raise FileNotFoundError(path)

    records: list[ModelT] = []

    with path.open("rb") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                records.append(model.model_validate_json(line))
            except ValidationError as error:
                raise ValueError(f"{path} line {line_number}: invalid record.") from error

    if not records:
        raise ValueError(f"Normalized artifact is empty: {path}.")

    return tuple(records)


def _load_and_validate_manifest(root: Path, *, dataset: DatasetName) -> dict[str, Any]:
    path = root / "manifest.json"

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(path) from None
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Invalid dataset manifest: {path}.") from error

    if not isinstance(payload, dict) or payload.get("dataset") != dataset.value:
        raise ValueError(f"Dataset manifest must identify {dataset.value!r}: {path}.")

    return cast(dict[str, Any], payload)


def _validate_document_group_isolation(loaded_splits: tuple[_LoadedSplit, ...]) -> None:
    groups_by_split: dict[DatasetSplit, set[str]] = defaultdict(set)

    for loaded in loaded_splits:
        documents_by_id = {document.document_id: document for document in loaded.documents}

        for example in loaded.examples:
            groups_by_split[loaded.split].add(
                _document_group_id(documents_by_id[example.question.document_id], example)
            )

    ordered = sorted(groups_by_split, key=lambda item: item.value)

    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            overlap = groups_by_split[left] & groups_by_split[right]

            if overlap:
                raise ValueError(
                    "Source-report groups overlap between "
                    f"{left.value} and {right.value}: {len(overlap)}."
                )


def _document_group_id(document: Document, example: DatasetExample) -> str:
    source_file = example.question.metadata.get("source_file")
    source = source_file if isinstance(source_file, str) else document.source_uri

    if source is None:
        source = document.document_id

    normalized = source.replace("\\", "/").strip().casefold()
    path = PurePosixPath(normalized)

    if len(path.parts) >= 3 and path.name.casefold().startswith("page_"):
        normalized = path.parent.as_posix()

    return f"source-group:{_stable_digest(normalized)}"


def _assigned_document_split(document_group_id: str) -> DatasetSplit:
    bucket = int(sha256(document_group_id.encode("utf-8")).hexdigest()[:8], 16) % 100

    if bucket < 80:
        return DatasetSplit.TRAIN

    if bucket < 90:
        return DatasetSplit.VALIDATION

    return DatasetSplit.TEST


def _validate_records(records: list[dict[str, object]], *, split: DatasetSplit) -> None:
    if not records:
        raise ValueError(f"Cannot export an empty RAG training split: {split.value}.")

    example_ids = tuple(cast(str, record["example_id"]) for record in records)

    if len(set(example_ids)) != len(example_ids):
        raise ValueError(f"Duplicate RAG training example IDs in {split.value}.")

    for record in records:
        messages = cast(list[dict[str, str]], record["messages"])

        if [message["role"] for message in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"Invalid message roles for {record['example_id']!r}.")

        if messages[0]["content"] != SYSTEM_INSTRUCTION:
            raise ValueError(f"Production system instruction changed for {record['example_id']!r}.")

        if cast(bool, record["answerable"]):
            source_numbers = cast(list[int], record["gold_source_numbers"])

            if not source_numbers or any(
                f"[Source {number}]" not in messages[2]["content"] for number in source_numbers
            ):
                raise ValueError(
                    f"Supported target lacks gold citations: {record['example_id']!r}."
                )
        elif messages[2]["content"] != UNSUPPORTED_ANSWER:
            raise ValueError(f"Unsupported target is not exact: {record['example_id']!r}.")


def _refusal_target(*, supported_count: int, refusal_ratio: float) -> int:
    if refusal_ratio == 0.0:
        return 0

    return round((supported_count * refusal_ratio) / (1.0 - refusal_ratio))


def _serialize_jsonl(records: Iterable[dict[str, object]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for record in records
    )


def _count_record_field(
    records: Iterable[dict[str, object]],
    field: str,
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)

    for record in records:
        counts[cast(str, record[field])] += 1

    return dict(sorted(counts.items()))


def _validate_splits(splits: Sequence[DatasetSplit]) -> tuple[DatasetSplit, ...]:
    materialized = tuple(splits)

    if not materialized:
        raise ValueError("At least one split must be selected.")

    if len(set(materialized)) != len(materialized):
        raise ValueError("Split selection contains duplicates.")

    return tuple(sorted(materialized, key=lambda item: item.value))


def _hash_elements(elements: Iterable[DocumentElement]) -> str:
    digest = sha256()

    for element in sorted(elements, key=lambda item: item.element_id):
        digest.update(element.element_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(element.source_text.encode("utf-8"))
        digest.update(b"\0")

    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _stable_digest(*values: str | None) -> str:
    digest = sha256()

    for value in values:
        encoded = (value or "").encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)

    return digest.hexdigest()[:24]
