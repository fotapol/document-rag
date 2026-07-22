"""Streaming integrity validation for prepared DocFinQA artifacts."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from document_rag.datasets.models import DatasetSplit


@dataclass(frozen=True, slots=True)
class DocFinQASplitIntegritySnapshot:
    """Identifiers collected from one prepared DocFinQA split."""

    split: DatasetSplit
    document_ids: frozenset[str]
    element_ids: frozenset[str]
    example_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class DocFinQADocumentOverlap:
    """Documents shared by two official DocFinQA splits."""

    left_split: DatasetSplit
    right_split: DatasetSplit
    document_ids: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.document_ids)


@dataclass(frozen=True, slots=True)
class DocFinQAIntegrityReport:
    """Successful DocFinQA integrity-validation result."""

    splits: tuple[DocFinQASplitIntegritySnapshot, ...]
    document_overlaps: tuple[DocFinQADocumentOverlap, ...]


@dataclass(frozen=True, slots=True)
class _ArtifactSpec:
    relative_path: str
    record_count: int
    checksum: str


_RecordValidator = Callable[[dict[str, Any], int], None]


def validate_docfinqa_output(
    output_directory: Path,
) -> DocFinQAIntegrityReport:
    """Validate manifest, checksums, JSONL records and references.

    Exact examples may not occur in multiple splits. Shared documents
    are reported because they are present in the official DocFinQA
    split assignment.
    """

    root = output_directory.resolve()
    manifest_path = root / "manifest.json"

    manifest = _load_json_object(
        manifest_path,
        entity_name="DocFinQA manifest",
    )

    if manifest.get("dataset") != "docfinqa":
        raise ValueError("Manifest dataset must be 'docfinqa'")

    split_payloads = _require_mapping(
        manifest.get("splits"),
        label="manifest splits",
    )

    if not split_payloads:
        raise ValueError("DocFinQA manifest contains no splits")

    snapshots: list[DocFinQASplitIntegritySnapshot] = []

    for split_name in sorted(split_payloads):
        try:
            split = DatasetSplit(split_name)
        except ValueError as error:
            raise ValueError(f"Unknown DocFinQA split: {split_name!r}") from error

        split_payload = _require_mapping(
            split_payloads[split_name],
            label=f"{split_name} split",
        )

        snapshots.append(
            _validate_split(
                root=root,
                split=split,
                payload=split_payload,
            )
        )

    overlaps = _validate_cross_split_integrity(snapshots)

    return DocFinQAIntegrityReport(
        splits=tuple(snapshots),
        document_overlaps=overlaps,
    )


def _validate_split(
    *,
    root: Path,
    split: DatasetSplit,
    payload: dict[str, Any],
) -> DocFinQASplitIntegritySnapshot:
    artifacts = _require_mapping(
        payload.get("artifacts"),
        label=f"{split.value} artifacts",
    )

    document_ids: set[str] = set()
    element_ids: set[str] = set()
    example_ids: set[str] = set()
    element_positions: set[tuple[str, int]] = set()

    document_spec = _read_artifact_spec(
        artifacts,
        name="documents",
    )

    def validate_document(
        record: dict[str, Any],
        line_number: int,
    ) -> None:
        _require_identity(
            record=record,
            split=split,
            line_number=line_number,
        )

        document_id = _require_string(
            record,
            "document_id",
            line_number,
        )

        if document_id in document_ids:
            raise ValueError(
                f"{split.value} documents line {line_number}: duplicate document ID {document_id!r}"
            )

        document_ids.add(document_id)

    _validate_jsonl_artifact(
        root=root,
        spec=document_spec,
        entity_name=f"{split.value} documents",
        validate_record=validate_document,
    )

    element_spec = _read_artifact_spec(
        artifacts,
        name="elements",
    )

    def validate_element(
        record: dict[str, Any],
        line_number: int,
    ) -> None:
        element_id = _require_string(
            record,
            "element_id",
            line_number,
        )
        document_id = _require_string(
            record,
            "document_id",
            line_number,
        )
        index = _require_integer(
            record,
            "index",
            line_number,
        )
        start_char = _require_integer(
            record,
            "start_char",
            line_number,
        )
        end_char = _require_integer(
            record,
            "end_char",
            line_number,
        )
        source_text = _require_string(
            record,
            "source_text",
            line_number,
            allow_whitespace=True,
        )

        if document_id not in document_ids:
            raise ValueError(
                f"{split.value} elements line {line_number}: unknown document {document_id!r}"
            )

        if element_id in element_ids:
            raise ValueError(
                f"{split.value} elements line {line_number}: duplicate element ID {element_id!r}"
            )

        if index < 0:
            raise ValueError(f"{split.value} elements line {line_number}: negative chunk index")

        position = (document_id, index)

        if position in element_positions:
            raise ValueError(
                f"{split.value} elements line "
                f"{line_number}: duplicate chunk index "
                f"{index} for document {document_id!r}"
            )

        if start_char < 0 or end_char <= start_char:
            raise ValueError(
                f"{split.value} elements line {line_number}: invalid character offsets"
            )

        if len(source_text) != end_char - start_char:
            raise ValueError(
                f"{split.value} elements line "
                f"{line_number}: source text length does "
                "not match character offsets"
            )

        element_ids.add(element_id)
        element_positions.add(position)

    _validate_jsonl_artifact(
        root=root,
        spec=element_spec,
        entity_name=f"{split.value} elements",
        validate_record=validate_element,
    )

    example_spec = _read_artifact_spec(
        artifacts,
        name="examples",
    )

    def validate_example(
        record: dict[str, Any],
        line_number: int,
    ) -> None:
        _require_identity(
            record=record,
            split=split,
            line_number=line_number,
        )

        example_id = _require_string(
            record,
            "example_id",
            line_number,
        )
        document_id = _require_string(
            record,
            "document_id",
            line_number,
        )

        _require_string(
            record,
            "question",
            line_number,
        )
        _require_string(
            record,
            "answer",
            line_number,
        )

        if example_id in example_ids:
            raise ValueError(
                f"{split.value} examples line {line_number}: duplicate example ID {example_id!r}"
            )

        if document_id not in document_ids:
            raise ValueError(
                f"{split.value} examples line {line_number}: unknown document {document_id!r}"
            )

        if record.get("link_status") not in {
            "exact",
            "equivalent",
        }:
            raise ValueError(
                f"{split.value} examples line {line_number}: invalid accepted link status"
            )

        supporting_facts = record.get("supporting_facts")

        if not isinstance(supporting_facts, list) or not supporting_facts:
            raise ValueError(
                f"{split.value} examples line "
                f"{line_number}: supporting facts "
                "must be a non-empty list"
            )

        for fact_index, raw_fact in enumerate(supporting_facts):
            fact = _require_mapping(
                raw_fact,
                label=(f"{split.value} example line {line_number} supporting fact {fact_index}"),
            )

            element_id = _require_string(
                fact,
                "element_id",
                line_number,
            )
            _require_string(
                fact,
                "source_key",
                line_number,
            )

            score = fact.get("score")

            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not 0.0 <= float(score) <= 1.0
            ):
                raise ValueError(
                    f"{split.value} examples line {line_number}: invalid evidence score"
                )

            if element_id not in element_ids:
                raise ValueError(
                    f"{split.value} examples line "
                    f"{line_number}: unknown supporting "
                    f"element {element_id!r}"
                )

        example_ids.add(example_id)

    _validate_jsonl_artifact(
        root=root,
        spec=example_spec,
        entity_name=f"{split.value} examples",
        validate_record=validate_example,
    )

    _validate_statistics(
        split=split,
        payload=payload,
        document_count=len(document_ids),
        example_count=len(example_ids),
    )

    return DocFinQASplitIntegritySnapshot(
        split=split,
        document_ids=frozenset(document_ids),
        element_ids=frozenset(element_ids),
        example_ids=frozenset(example_ids),
    )


def _validate_cross_split_integrity(
    snapshots: list[DocFinQASplitIntegritySnapshot],
) -> tuple[DocFinQADocumentOverlap, ...]:
    overlaps: list[DocFinQADocumentOverlap] = []

    for index, snapshot in enumerate(snapshots):
        for previous in snapshots[:index]:
            example_overlap = snapshot.example_ids & previous.example_ids

            if example_overlap:
                raise ValueError(
                    f"Splits {previous.split.value!r} and "
                    f"{snapshot.split.value!r} share "
                    f"{len(example_overlap)} example IDs"
                )

            document_overlap = snapshot.document_ids & previous.document_ids

            if document_overlap:
                overlaps.append(
                    DocFinQADocumentOverlap(
                        left_split=previous.split,
                        right_split=snapshot.split,
                        document_ids=tuple(sorted(document_overlap)),
                    )
                )

    return tuple(overlaps)


def _validate_statistics(
    *,
    split: DatasetSplit,
    payload: dict[str, Any],
    document_count: int,
    example_count: int,
) -> None:
    stats = _require_mapping(
        payload.get("statistics"),
        label=f"{split.value} statistics",
    )

    total_records = _statistics_integer(
        stats,
        "total_records",
        split,
    )
    normalized_records = _statistics_integer(
        stats,
        "normalized_records",
        split,
    )
    skipped_records = _statistics_integer(
        stats,
        "skipped_records",
        split,
    )
    unique_documents = _statistics_integer(
        stats,
        "unique_documents",
        split,
    )

    reason_total = sum(
        _statistics_integer(
            stats,
            key,
            split,
        )
        for key in stats
        if (key.startswith("skipped_") and key != "skipped_records")
    )

    if total_records != (normalized_records + skipped_records):
        raise ValueError(f"{split.value} statistics do not add up")

    if skipped_records != reason_total:
        raise ValueError(f"{split.value} skipped-record statistics do not add up")

    if normalized_records != example_count:
        raise ValueError(f"{split.value} normalized record count does not match examples")

    if unique_documents != document_count:
        raise ValueError(f"{split.value} unique document count does not match documents")


def _validate_jsonl_artifact(
    *,
    root: Path,
    spec: _ArtifactSpec,
    entity_name: str,
    validate_record: _RecordValidator,
) -> None:
    relative_path = Path(spec.relative_path)

    if relative_path.is_absolute():
        raise ValueError(f"{entity_name} path must be relative")

    path = (root / relative_path).resolve()

    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{entity_name} path escapes output directory") from error

    if not path.is_file():
        raise FileNotFoundError(path)

    digest = sha256()
    record_count = 0

    with path.open("rb") as artifact_file:
        for line_number, raw_line in enumerate(
            artifact_file,
            start=1,
        ):
            digest.update(raw_line)
            record_count += 1

            try:
                raw_record = json.loads(raw_line)
            except (
                json.JSONDecodeError,
                UnicodeDecodeError,
            ) as error:
                raise ValueError(f"{entity_name} line {line_number}: invalid JSON") from error

            if not isinstance(raw_record, dict):
                raise ValueError(f"{entity_name} line {line_number}: record must be an object")

            validate_record(
                cast(dict[str, Any], raw_record),
                line_number,
            )

    if record_count != spec.record_count:
        raise ValueError(
            f"{entity_name} record count mismatch: "
            f"expected {spec.record_count}, "
            f"found {record_count}"
        )

    actual_checksum = digest.hexdigest()

    if actual_checksum != spec.checksum:
        raise ValueError(f"{entity_name} SHA-256 mismatch")


def _read_artifact_spec(
    artifacts: dict[str, Any],
    *,
    name: str,
) -> _ArtifactSpec:
    payload = _require_mapping(
        artifacts.get(name),
        label=f"{name} artifact",
    )

    relative_path = payload.get("path")
    record_count = payload.get("record_count")
    checksum = payload.get("sha256")

    if not isinstance(relative_path, str):
        raise ValueError(f"{name} artifact path must be a string")

    if isinstance(record_count, bool) or not isinstance(record_count, int) or record_count <= 0:
        raise ValueError(f"{name} artifact record count must be positive")

    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
    ):
        raise ValueError(f"{name} artifact SHA-256 is invalid")

    return _ArtifactSpec(
        relative_path=relative_path,
        record_count=record_count,
        checksum=checksum,
    )


def _require_identity(
    *,
    record: dict[str, Any],
    split: DatasetSplit,
    line_number: int,
) -> None:
    if record.get("dataset") != "docfinqa":
        raise ValueError(f"{split.value} line {line_number}: dataset must be 'docfinqa'")

    if record.get("split") != split.value:
        raise ValueError(f"{split.value} line {line_number}: split value does not match artifact")


def _require_string(
    record: dict[str, Any],
    key: str,
    line_number: int,
    *,
    allow_whitespace: bool = False,
) -> str:
    value = record.get(key)

    if not isinstance(value, str):
        raise ValueError(f"Line {line_number}: {key!r} must be a string")

    if not allow_whitespace and not value.strip():
        raise ValueError(f"Line {line_number}: {key!r} cannot be empty")

    return value


def _require_integer(
    record: dict[str, Any],
    key: str,
    line_number: int,
) -> int:
    value = record.get(key)

    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise ValueError(f"Line {line_number}: {key!r} must be an integer")

    return value


def _statistics_integer(
    stats: dict[str, Any],
    key: str,
    split: DatasetSplit,
) -> int:
    value = stats.get(key)

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{split.value} statistic {key!r} must be a non-negative integer")

    return value


def _require_mapping(
    value: object,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")

    return cast(dict[str, Any], value)


def _load_json_object(
    path: Path,
    *,
    entity_name: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as error:
        raise ValueError(f"{entity_name} contains invalid JSON") from error

    return _require_mapping(
        payload,
        label=entity_name,
    )
