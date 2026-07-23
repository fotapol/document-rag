from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol

from document_rag.datasets.models import DatasetExample, DatasetSplit
from document_rag.domain import Document, DocumentElement


class PreparedDatasetSplit(Protocol):
    """Normalized dataset split required by integrity validation."""

    @property
    def split(self) -> DatasetSplit: ...

    @property
    def documents(self) -> tuple[Document, ...]: ...

    @property
    def elements(self) -> tuple[DocumentElement, ...]: ...

    @property
    def examples(self) -> tuple[DatasetExample, ...]: ...


@dataclass(frozen=True, slots=True)
class SplitIntegritySnapshot:
    """Identifiers collected from one normalized dataset split."""

    split: DatasetSplit
    example_ids: frozenset[str]
    document_ids: frozenset[str]
    element_ids: frozenset[str]
    report_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class ReportOverlap:
    """Reports shared by two otherwise disjoint dataset splits."""

    left_split: DatasetSplit
    right_split: DatasetSplit
    report_ids: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.report_ids)


def build_split_integrity_snapshot(
    prepared_split: PreparedDatasetSplit,
) -> SplitIntegritySnapshot:
    """Validate internal references and collect split identifiers."""

    example_ids = [example.example_id for example in prepared_split.examples]
    document_ids = [document.document_id for document in prepared_split.documents]
    element_ids = [element.element_id for element in prepared_split.elements]

    _require_unique(example_ids, entity_name="example")
    _require_unique(document_ids, entity_name="document")
    _require_unique(element_ids, entity_name="document element")

    document_id_set = frozenset(document_ids)
    element_id_set = frozenset(element_ids)

    for element in prepared_split.elements:
        if element.document_id not in document_id_set:
            raise ValueError(
                f"Element {element.element_id!r} references unknown "
                f"document {element.document_id!r}"
            )

    for example in prepared_split.examples:
        if example.question.document_id not in document_id_set:
            raise ValueError(
                f"Example {example.example_id!r} references unknown "
                f"document {example.question.document_id!r}"
            )

        for supporting_fact in example.supporting_facts:
            if supporting_fact.element_id not in element_id_set:
                raise ValueError(
                    f"Example {example.example_id!r} references unknown "
                    f"element {supporting_fact.element_id!r}"
                )

    return SplitIntegritySnapshot(
        split=prepared_split.split,
        example_ids=frozenset(example_ids),
        document_ids=document_id_set,
        element_ids=element_id_set,
        report_ids=frozenset(_report_id(document) for document in prepared_split.documents),
    )


def validate_cross_split_integrity(
    snapshot: SplitIntegritySnapshot,
    *,
    previous_snapshots: Sequence[SplitIntegritySnapshot],
) -> tuple[ReportOverlap, ...]:
    """Reject exact split overlap and report shared annual reports."""

    report_overlaps: list[ReportOverlap] = []

    for previous in previous_snapshots:
        example_overlap = snapshot.example_ids & previous.example_ids

        if example_overlap:
            raise ValueError(
                f"Splits {previous.split.value!r} and "
                f"{snapshot.split.value!r} share "
                f"{len(example_overlap)} example IDs"
            )

        document_overlap = snapshot.document_ids & previous.document_ids

        if document_overlap:
            raise ValueError(
                f"Splits {previous.split.value!r} and "
                f"{snapshot.split.value!r} share "
                f"{len(document_overlap)} documents"
            )

        report_overlap = snapshot.report_ids & previous.report_ids

        if report_overlap:
            report_overlaps.append(
                ReportOverlap(
                    left_split=previous.split,
                    right_split=snapshot.split,
                    report_ids=tuple(sorted(report_overlap)),
                )
            )

    return tuple(report_overlaps)


def _require_unique(
    identifiers: Sequence[str],
    *,
    entity_name: str,
) -> None:
    duplicate_count = len(identifiers) - len(set(identifiers))

    if duplicate_count:
        raise ValueError(f"Split contains {duplicate_count} duplicate {entity_name} IDs")


def _report_id(document: Document) -> str:
    if document.source_uri:
        source_path = PurePosixPath(document.source_uri)

        if len(source_path.parts) >= 2:
            return "/".join(source_path.parts[:-1])

    return document.document_id
