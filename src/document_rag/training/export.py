"""Export normalized FinQA and DocFinQA data to deterministic chat JSONL."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from document_rag.datasets.models import DatasetName, DatasetSplit

SYSTEM_PROMPT = (
    "Answer the financial question using only the provided context. Return only the final answer."
)


@dataclass(frozen=True, slots=True)
class ExportedTrainingArtifact:
    """Metadata for one exported split artifact."""

    split: DatasetSplit
    path: Path
    record_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class FinancialQAExportResult:
    """Complete financial QA training-data export result."""

    artifacts: tuple[ExportedTrainingArtifact, ...]
    manifest_path: Path
    manifest_sha256: str


def export_financial_qa_training_data(
    *,
    finqa_directory: Path,
    docfinqa_directory: Path,
    output_directory: Path,
    splits: Sequence[DatasetSplit] = tuple(DatasetSplit),
) -> FinancialQAExportResult:
    """Export FinQA and DocFinQA examples to chat-format JSONL."""

    selected_splits = _validate_splits(splits)
    finqa_root = finqa_directory.resolve()
    docfinqa_root = docfinqa_directory.resolve()
    output_root = output_directory.resolve()

    finqa_manifest = _load_json_object(
        finqa_root / "manifest.json",
        label="FinQA manifest",
    )
    docfinqa_manifest = _load_json_object(
        docfinqa_root / "manifest.json",
        label="DocFinQA manifest",
    )

    _validate_manifest_dataset(
        manifest=finqa_manifest,
        expected=DatasetName.FINQA,
        label="FinQA manifest",
    )
    _validate_manifest_dataset(
        manifest=docfinqa_manifest,
        expected=DatasetName.DOCFINQA,
        label="DocFinQA manifest",
    )

    output_root.mkdir(parents=True, exist_ok=True)

    artifacts: list[ExportedTrainingArtifact] = []
    split_example_ids: dict[DatasetSplit, set[str]] = {}

    for split in selected_splits:
        records = [
            *_iter_training_records(
                dataset=DatasetName.FINQA,
                dataset_directory=finqa_root,
                split=split,
            ),
            *_iter_training_records(
                dataset=DatasetName.DOCFINQA,
                dataset_directory=docfinqa_root,
                split=split,
            ),
        ]
        records.sort(
            key=lambda record: (
                cast(str, record["dataset"]),
                cast(str, record["example_id"]),
            )
        )

        example_ids = {cast(str, record["example_id"]) for record in records}

        if len(example_ids) != len(records):
            raise ValueError(f"Duplicate example IDs found in {split.value} export")

        split_example_ids[split] = example_ids

        artifacts.append(
            _write_jsonl(
                path=output_root / f"{split.value}.jsonl",
                split=split,
                records=records,
            )
        )

    _validate_cross_split_ids(split_example_ids)

    manifest_payload = {
        "artifacts": {
            artifact.split.value: {
                "path": artifact.path.relative_to(output_root).as_posix(),
                "record_count": artifact.record_count,
                "sha256": artifact.sha256,
            }
            for artifact in artifacts
        },
        "format": "chat_jsonl",
        "inputs": {
            DatasetName.FINQA.value: {
                "manifest_sha256": _sha256_file(finqa_root / "manifest.json"),
                "source": finqa_manifest.get("source"),
                "sources": finqa_manifest.get("sources"),
            },
            DatasetName.DOCFINQA.value: {
                "manifest_sha256": _sha256_file(docfinqa_root / "manifest.json"),
                "source": docfinqa_manifest.get("source"),
                "sources": docfinqa_manifest.get("sources"),
            },
        },
        "schema_version": "1",
        "system_prompt": SYSTEM_PROMPT,
    }

    manifest_bytes = (
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
    _write_bytes_atomically(
        path=manifest_path,
        content=manifest_bytes,
    )

    return FinancialQAExportResult(
        artifacts=tuple(artifacts),
        manifest_path=manifest_path,
        manifest_sha256=sha256(manifest_bytes).hexdigest(),
    )


def _iter_training_records(
    *,
    dataset: DatasetName,
    dataset_directory: Path,
    split: DatasetSplit,
) -> Iterator[dict[str, Any]]:
    split_directory = dataset_directory / split.value
    elements_path = split_directory / "elements.jsonl"
    examples_path = split_directory / "examples.jsonl"

    elements = _load_elements(elements_path)

    for line_number, example in _iter_json_objects(examples_path):
        example_dataset = _require_string(
            example,
            "dataset",
            path=examples_path,
            line_number=line_number,
        )

        if example_dataset != dataset.value:
            raise ValueError(
                f"{examples_path} line {line_number}: expected dataset "
                f"{dataset.value!r}, got {example_dataset!r}"
            )

        example_split = _require_string(
            example,
            "split",
            path=examples_path,
            line_number=line_number,
        )

        if example_split != split.value:
            raise ValueError(
                f"{examples_path} line {line_number}: expected split "
                f"{split.value!r}, got {example_split!r}"
            )

        example_id = _require_string(
            example,
            "example_id",
            path=examples_path,
            line_number=line_number,
        )
        question = _extract_question(
            dataset=dataset,
            example=example,
            path=examples_path,
            line_number=line_number,
        )
        answer = _extract_answer(
            dataset=dataset,
            example=example,
            path=examples_path,
            line_number=line_number,
        )
        context = _build_context(
            example=example,
            elements=elements,
            path=examples_path,
            line_number=line_number,
        )

        yield {
            "dataset": dataset.value,
            "example_id": example_id,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (f"Context:\n{context}\n\nQuestion:\n{question}"),
                },
                {
                    "role": "assistant",
                    "content": answer,
                },
            ],
            "split": split.value,
        }


def _load_elements(path: Path) -> dict[str, str]:
    elements: dict[str, str] = {}

    for line_number, element in _iter_json_objects(path):
        element_id = _require_string(
            element,
            "element_id",
            path=path,
            line_number=line_number,
        )
        source_text = _require_string(
            element,
            "source_text",
            path=path,
            line_number=line_number,
        )

        if element_id in elements:
            raise ValueError(f"{path} line {line_number}: duplicate element ID {element_id!r}")

        elements[element_id] = source_text

    if not elements:
        raise ValueError(f"No elements found in {path}")

    return elements


def _extract_question(
    *,
    dataset: DatasetName,
    example: dict[str, Any],
    path: Path,
    line_number: int,
) -> str:
    raw_question = example.get("question")

    if isinstance(raw_question, dict):
        question = _require_mapping(
            raw_question,
            label=f"{path} line {line_number} question",
        )
        return _require_string(
            question,
            "text",
            path=path,
            line_number=line_number,
        )

    if (
        dataset is not DatasetName.DOCFINQA
        or not isinstance(raw_question, str)
        or not raw_question.strip()
    ):
        raise ValueError(
            f"{path} line {line_number}: question must be a normalized question object"
        )

    return raw_question.strip()


def _extract_answer(
    *,
    dataset: DatasetName,
    example: dict[str, Any],
    path: Path,
    line_number: int,
) -> str:
    raw_reference_answer = example.get("reference_answer")

    if isinstance(raw_reference_answer, dict):
        reference_answer = _require_mapping(
            raw_reference_answer,
            label=f"{path} line {line_number} reference_answer",
        )
        return _require_string(
            reference_answer,
            "text",
            path=path,
            line_number=line_number,
        )

    if dataset is not DatasetName.DOCFINQA:
        raise ValueError(
            f"{path} line {line_number}: reference_answer must be a normalized answer object"
        )

    return _require_string(
        example,
        "answer",
        path=path,
        line_number=line_number,
    )


def _build_context(
    *,
    example: dict[str, Any],
    elements: dict[str, str],
    path: Path,
    line_number: int,
) -> str:
    raw_facts = example.get("supporting_facts")

    if not isinstance(raw_facts, list) or not raw_facts:
        raise ValueError(f"{path} line {line_number}: supporting_facts must be a non-empty list")

    context_parts: list[str] = []
    seen_element_ids: set[str] = set()

    for fact_index, raw_fact in enumerate(raw_facts):
        fact = _require_mapping(
            raw_fact,
            label=(f"{path} line {line_number} supporting fact {fact_index}"),
        )
        element_id = _require_string(
            fact,
            "element_id",
            path=path,
            line_number=line_number,
        )
        source_key = _require_string(
            fact,
            "source_key",
            path=path,
            line_number=line_number,
        )

        if element_id in seen_element_ids:
            continue

        try:
            source_text = elements[element_id]
        except KeyError as error:
            raise ValueError(
                f"{path} line {line_number}: supporting fact references "
                f"unknown element {element_id!r}"
            ) from error

        seen_element_ids.add(element_id)
        context_parts.append(f"[{source_key}]\n{source_text}")

    if not context_parts:
        raise ValueError(f"{path} line {line_number}: no usable supporting context")

    return "\n\n".join(context_parts)


def _write_jsonl(
    *,
    path: Path,
    split: DatasetSplit,
    records: Iterable[dict[str, Any]],
) -> ExportedTrainingArtifact:
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
                line = (
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")

                temporary_file.write(line)
                digest.update(line)
                record_count += 1

            temporary_file.flush()
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    if record_count == 0:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(f"Cannot export empty split: {split.value}")

    try:
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return ExportedTrainingArtifact(
        split=split,
        path=path,
        record_count=record_count,
        sha256=digest.hexdigest(),
    )


def _validate_cross_split_ids(
    split_example_ids: dict[DatasetSplit, set[str]],
) -> None:
    ordered_splits = sorted(
        split_example_ids,
        key=lambda split: split.value,
    )

    for left_index, left_split in enumerate(ordered_splits):
        for right_split in ordered_splits[left_index + 1 :]:
            overlap = split_example_ids[left_split] & split_example_ids[right_split]

            if overlap:
                raise ValueError(
                    "Training export example IDs overlap between "
                    f"{left_split.value} and {right_split.value}: "
                    f"{len(overlap)}"
                )


def _validate_splits(
    splits: Sequence[DatasetSplit],
) -> tuple[DatasetSplit, ...]:
    selected_splits = tuple(splits)

    if not selected_splits:
        raise ValueError("At least one split must be selected")

    if len(set(selected_splits)) != len(selected_splits):
        raise ValueError("Split selection contains duplicates")

    return tuple(
        sorted(
            selected_splits,
            key=lambda split: split.value,
        )
    )


def _validate_manifest_dataset(
    *,
    manifest: dict[str, Any],
    expected: DatasetName,
    label: str,
) -> None:
    actual = manifest.get("dataset")

    if actual != expected.value:
        raise ValueError(f"{label} dataset must be {expected.value!r}, got {actual!r}")


def _iter_json_objects(
    path: Path,
) -> Iterator[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("rb") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            try:
                payload = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError(f"{path} line {line_number}: invalid JSON") from error

            if not isinstance(payload, dict):
                raise ValueError(f"{path} line {line_number}: record must be an object")

            yield line_number, cast(dict[str, Any], payload)


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


def _require_string(
    record: dict[str, Any],
    key: str,
    *,
    path: Path,
    line_number: int,
) -> str:
    value = record.get(key)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} line {line_number}: {key!r} must be a non-empty string")

    return value.strip()


def _sha256_file(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _write_bytes_atomically(
    *,
    path: Path,
    content: bytes,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

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
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
