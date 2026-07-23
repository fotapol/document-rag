"""Deterministic development subset generation for DocFinQA."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from document_rag.datasets.config import DatasetConfig
from document_rag.datasets.docfinqa.manifest import (
    DocFinQASplitManifestInput,
    WrittenDocFinQAManifest,
    write_docfinqa_manifest,
)
from document_rag.datasets.docfinqa.preparer import DocFinQAPreparationStats
from document_rag.datasets.docfinqa.writer import (
    WrittenDocFinQAArtifact,
    WrittenDocFinQASplit,
)
from document_rag.datasets.models import DatasetName, DatasetSplit

DEFAULT_SAMPLE_DOCUMENTS_PER_SPLIT = 5


@dataclass(frozen=True, slots=True)
class DocFinQASampleResult:
    """Artifacts generated for the DocFinQA development subset."""

    splits: tuple[WrittenDocFinQASplit, ...]
    manifest: WrittenDocFinQAManifest


def create_docfinqa_sample(
    *,
    input_directory: Path,
    output_directory: Path,
    config: DatasetConfig,
    splits: Sequence[DatasetSplit],
    chunk_size: int,
    chunk_overlap: int,
    evidence_minimum_score: float,
    documents_per_split: int = DEFAULT_SAMPLE_DOCUMENTS_PER_SPLIT,
) -> DocFinQASampleResult:
    """Create a deterministic, referentially complete DocFinQA subset."""

    if config.name is not DatasetName.DOCFINQA:
        raise ValueError("DocFinQA sample requires a DocFinQA configuration")

    if documents_per_split <= 0:
        raise ValueError("Sample documents per split must be positive")

    selected_splits = _validate_splits(splits)
    input_root = input_directory.resolve()
    output_root = output_directory.resolve()

    if input_root == output_root:
        raise ValueError("Sample output directory must differ from its input")

    source_manifest = _load_json_object(
        input_root / "manifest.json",
        label="DocFinQA source manifest",
    )

    if source_manifest.get("dataset") != DatasetName.DOCFINQA.value:
        raise ValueError("Source manifest dataset must be 'docfinqa'")

    source_splits = _require_mapping(
        source_manifest.get("splits"),
        label="source manifest splits",
    )

    output_root.mkdir(parents=True, exist_ok=True)

    written_splits: list[WrittenDocFinQASplit] = []
    manifest_inputs: list[DocFinQASplitManifestInput] = []

    for split in selected_splits:
        split_payload = _require_mapping(
            source_splits.get(split.value),
            label=f"{split.value} source split",
        )
        artifacts = _require_mapping(
            split_payload.get("artifacts"),
            label=f"{split.value} source artifacts",
        )

        documents_path = _artifact_path(
            root=input_root,
            artifacts=artifacts,
            name="documents",
        )
        elements_path = _artifact_path(
            root=input_root,
            artifacts=artifacts,
            name="elements",
        )
        examples_path = _artifact_path(
            root=input_root,
            artifacts=artifacts,
            name="examples",
        )

        selected_documents = _select_documents(
            path=documents_path,
            split=split,
            limit=documents_per_split,
        )
        selected_document_ids = tuple(
            _require_nonempty_string(
                record,
                "document_id",
                path=documents_path,
                line_number=0,
            )
            for record in selected_documents
        )
        selected_document_id_set = set(selected_document_ids)
        document_order = {
            document_id: index for index, document_id in enumerate(selected_document_ids)
        }

        selected_elements = _select_elements(
            path=elements_path,
            selected_document_ids=selected_document_id_set,
        )
        selected_examples = _select_examples(
            path=examples_path,
            split=split,
            selected_document_ids=selected_document_id_set,
        )

        _validate_selected_records(
            split=split,
            document_ids=selected_document_ids,
            elements=selected_elements,
            examples=selected_examples,
        )

        selected_elements.sort(
            key=lambda record: (
                document_order[cast(str, record["document_id"])],
                cast(int, record["index"]),
                cast(str, record["element_id"]),
            )
        )
        selected_examples.sort(
            key=lambda record: (
                document_order[cast(str, record["document_id"])],
                cast(str, record["example_id"]),
            )
        )

        split_directory = output_root / split.value
        documents_artifact = _write_jsonl_artifact(
            path=split_directory / "documents.jsonl",
            records=selected_documents,
        )
        elements_artifact = _write_jsonl_artifact(
            path=split_directory / "elements.jsonl",
            records=selected_elements,
        )
        examples_artifact = _write_jsonl_artifact(
            path=split_directory / "examples.jsonl",
            records=selected_examples,
        )

        written_split = WrittenDocFinQASplit(
            split=split,
            documents=documents_artifact,
            elements=elements_artifact,
            examples=examples_artifact,
        )

        exact_links = sum(record.get("link_status") == "exact" for record in selected_examples)
        equivalent_links = sum(
            record.get("link_status") == "equivalent" for record in selected_examples
        )

        stats = DocFinQAPreparationStats(
            total_records=len(selected_examples),
            normalized_records=len(selected_examples),
            unique_documents=len(selected_documents),
            exact_links=exact_links,
            equivalent_links=equivalent_links,
            skipped_ambiguous=0,
            skipped_answer_mismatch=0,
            skipped_evidence_incomplete=0,
            unmatched_evidence_facts=0,
            skipped_duplicate=0,
        )

        written_splits.append(written_split)
        manifest_inputs.append(
            DocFinQASplitManifestInput(
                written_split=written_split,
                stats=stats,
            )
        )

    manifest = write_docfinqa_manifest(
        output_directory=output_root,
        config=config,
        split_inputs=manifest_inputs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        evidence_minimum_score=evidence_minimum_score,
        sample_documents_per_split=documents_per_split,
    )

    return DocFinQASampleResult(
        splits=tuple(written_splits),
        manifest=manifest,
    )


def _select_documents(
    *,
    path: Path,
    split: DatasetSplit,
    limit: int,
) -> list[dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}

    for line_number, record in _iter_json_objects(path):
        if record.get("dataset") != DatasetName.DOCFINQA.value:
            raise ValueError(f"{path} line {line_number}: dataset must be 'docfinqa'")

        if record.get("split") != split.value:
            raise ValueError(f"{path} line {line_number}: split does not match artifact")

        document_id = _require_nonempty_string(
            record,
            "document_id",
            path=path,
            line_number=line_number,
        )

        if document_id in documents:
            raise ValueError(f"{path} line {line_number}: duplicate document ID {document_id!r}")

        documents[document_id] = record

    if not documents:
        raise ValueError(f"Cannot create sample from empty {split.value} document artifact")

    selected_ids = sorted(
        documents,
        key=_document_selection_key,
    )[:limit]

    return [documents[document_id] for document_id in selected_ids]


def _select_elements(
    *,
    path: Path,
    selected_document_ids: set[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_element_ids: set[str] = set()

    for line_number, record in _iter_json_objects(path):
        document_id = _require_nonempty_string(
            record,
            "document_id",
            path=path,
            line_number=line_number,
        )

        if document_id not in selected_document_ids:
            continue

        element_id = _require_nonempty_string(
            record,
            "element_id",
            path=path,
            line_number=line_number,
        )
        index = _require_integer(
            record,
            "index",
            path=path,
            line_number=line_number,
        )
        start_char = _require_integer(
            record,
            "start_char",
            path=path,
            line_number=line_number,
        )
        end_char = _require_integer(
            record,
            "end_char",
            path=path,
            line_number=line_number,
        )
        source_text = _require_string(
            record,
            "source_text",
            path=path,
            line_number=line_number,
        )

        if element_id in seen_element_ids:
            raise ValueError(f"{path} line {line_number}: duplicate element ID {element_id!r}")

        if index < 0:
            raise ValueError(f"{path} line {line_number}: negative element index")

        if start_char < 0 or end_char <= start_char:
            raise ValueError(f"{path} line {line_number}: invalid character offsets")

        if len(source_text) != end_char - start_char:
            raise ValueError(
                f"{path} line {line_number}: source text length does not match character offsets"
            )

        seen_element_ids.add(element_id)
        selected.append(record)

    return selected


def _select_examples(
    *,
    path: Path,
    split: DatasetSplit,
    selected_document_ids: set[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_example_ids: set[str] = set()

    for line_number, record in _iter_json_objects(path):
        document_id = _require_nonempty_string(
            record,
            "document_id",
            path=path,
            line_number=line_number,
        )

        if document_id not in selected_document_ids:
            continue

        if record.get("dataset") != DatasetName.DOCFINQA.value:
            raise ValueError(f"{path} line {line_number}: dataset must be 'docfinqa'")

        if record.get("split") != split.value:
            raise ValueError(f"{path} line {line_number}: split does not match artifact")

        example_id = _require_nonempty_string(
            record,
            "example_id",
            path=path,
            line_number=line_number,
        )
        _require_nonempty_string(
            record,
            "question",
            path=path,
            line_number=line_number,
        )
        _require_nonempty_string(
            record,
            "answer",
            path=path,
            line_number=line_number,
        )

        if example_id in seen_example_ids:
            raise ValueError(f"{path} line {line_number}: duplicate example ID {example_id!r}")

        link_status = record.get("link_status")

        if link_status not in {"exact", "equivalent"}:
            raise ValueError(f"{path} line {line_number}: invalid accepted link status")

        supporting_facts = record.get("supporting_facts")

        if not isinstance(supporting_facts, list) or not supporting_facts:
            raise ValueError(
                f"{path} line {line_number}: supporting_facts must be a non-empty list"
            )

        seen_example_ids.add(example_id)
        selected.append(record)

    return selected


def _validate_selected_records(
    *,
    split: DatasetSplit,
    document_ids: tuple[str, ...],
    elements: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> None:
    document_id_set = set(document_ids)
    element_ids = {cast(str, element["element_id"]) for element in elements}

    element_documents = {cast(str, element["document_id"]) for element in elements}
    example_documents = {cast(str, example["document_id"]) for example in examples}

    missing_element_documents = document_id_set - element_documents

    if missing_element_documents:
        raise ValueError(
            f"{split.value} sample documents without elements: {len(missing_element_documents)}"
        )

    missing_example_documents = document_id_set - example_documents

    if missing_example_documents:
        raise ValueError(
            f"{split.value} sample documents without examples: {len(missing_example_documents)}"
        )

    for example in examples:
        supporting_facts = cast(list[object], example["supporting_facts"])

        for fact_index, raw_fact in enumerate(supporting_facts):
            fact = _require_mapping(
                raw_fact,
                label=(
                    f"{split.value} example {example['example_id']!r} supporting fact {fact_index}"
                ),
            )
            element_id = fact.get("element_id")

            if not isinstance(element_id, str) or not element_id:
                raise ValueError(
                    f"{split.value} example {example['example_id']!r}: "
                    "supporting fact has invalid element ID"
                )

            if element_id not in element_ids:
                raise ValueError(
                    f"{split.value} example {example['example_id']!r}: "
                    f"unknown supporting element {element_id!r}"
                )

            score = fact.get("score")

            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not 0.0 <= float(score) <= 1.0
            ):
                raise ValueError(
                    f"{split.value} example {example['example_id']!r}: invalid evidence score"
                )


def _write_jsonl_artifact(
    *,
    path: Path,
    records: Iterable[dict[str, Any]],
) -> WrittenDocFinQAArtifact:
    path.parent.mkdir(parents=True, exist_ok=True)

    digest = sha256()
    record_count = 0

    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

        try:
            for record in records:
                serialized = json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                line = serialized + b"\n"

                temporary_file.write(line)
                digest.update(line)
                record_count += 1

            temporary_file.flush()
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    if record_count == 0:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(f"Cannot write empty sample artifact: {path.name}")

    try:
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return WrittenDocFinQAArtifact(
        path=path,
        record_count=record_count,
        sha256=digest.hexdigest(),
    )


def _document_selection_key(document_id: str) -> tuple[bytes, str]:
    return (
        sha256(document_id.encode("utf-8")).digest(),
        document_id,
    )


def _artifact_path(
    *,
    root: Path,
    artifacts: dict[str, Any],
    name: str,
) -> Path:
    artifact = _require_mapping(
        artifacts.get(name),
        label=f"{name} artifact",
    )
    relative_value = artifact.get("path")

    if not isinstance(relative_value, str):
        raise ValueError(f"{name} artifact path must be a string")

    relative_path = Path(relative_value)

    if relative_path.is_absolute():
        raise ValueError(f"{name} artifact path must be relative")

    resolved_path = (root / relative_path).resolve()

    try:
        resolved_path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} artifact path escapes the source directory") from error

    if not resolved_path.is_file():
        raise FileNotFoundError(resolved_path)

    return resolved_path


def _iter_json_objects(
    path: Path,
) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("rb") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            try:
                raw_payload = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError(f"{path} line {line_number}: invalid JSON") from error

            if not isinstance(raw_payload, dict):
                raise ValueError(f"{path} line {line_number}: record must be an object")

            yield line_number, cast(dict[str, Any], raw_payload)


def _validate_splits(
    splits: Sequence[DatasetSplit],
) -> tuple[DatasetSplit, ...]:
    selected_splits = tuple(splits)

    if not selected_splits:
        raise ValueError("At least one sample split must be selected")

    if len(set(selected_splits)) != len(selected_splits):
        raise ValueError("Sample split selection contains duplicates")

    return tuple(
        sorted(
            selected_splits,
            key=lambda split: split.value,
        )
    )


def _require_nonempty_string(
    record: dict[str, Any],
    key: str,
    *,
    path: Path,
    line_number: int,
) -> str:
    value = _require_string(
        record,
        key,
        path=path,
        line_number=line_number,
    )

    if not value.strip():
        raise ValueError(f"{path} line {line_number}: {key!r} cannot be empty")

    return value


def _require_string(
    record: dict[str, Any],
    key: str,
    *,
    path: Path,
    line_number: int,
) -> str:
    value = record.get(key)

    if not isinstance(value, str):
        raise ValueError(f"{path} line {line_number}: {key!r} must be a string")

    return value


def _require_integer(
    record: dict[str, Any],
    key: str,
    *,
    path: Path,
    line_number: int,
) -> int:
    value = record.get(key)

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} line {line_number}: {key!r} must be an integer")

    return value


def _load_json_object(
    path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{label} contains invalid JSON") from error

    return _require_mapping(payload, label=label)


def _require_mapping(
    value: object,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")

    return cast(dict[str, Any], value)
