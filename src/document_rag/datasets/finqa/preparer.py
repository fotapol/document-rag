from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from document_rag.datasets.finqa.normalizer import normalize_finqa_record
from document_rag.datasets.finqa.raw_models import FinQARawRecord
from document_rag.datasets.models import DatasetExample, DatasetSplit
from document_rag.domain import Document, DocumentElement


class FinQARecordSource(Protocol):
    """Minimal interface required to prepare a FinQA split."""

    def read_split(self, split: DatasetSplit) -> Sequence[FinQARawRecord]: ...


@dataclass(frozen=True, slots=True)
class PreparedFinQASplit:
    """Fully normalized and validated FinQA split."""

    split: DatasetSplit
    documents: tuple[Document, ...]
    elements: tuple[DocumentElement, ...]
    examples: tuple[DatasetExample, ...]


def prepare_finqa_split(
    source: FinQARecordSource,
    *,
    split: DatasetSplit,
) -> PreparedFinQASplit:
    """Normalize all records from one FinQA split."""

    documents_by_id: dict[str, Document] = {}
    elements_by_id: dict[str, DocumentElement] = {}
    examples_by_id: dict[str, DatasetExample] = {}

    for raw_record in source.read_split(split):
        normalized = normalize_finqa_record(raw_record, split=split)

        _store_reusable_entity(
            documents_by_id,
            entity_id=normalized.document.document_id,
            entity=normalized.document,
            entity_name="document",
        )

        for element in normalized.elements:
            _store_reusable_entity(
                elements_by_id,
                entity_id=element.element_id,
                entity=element,
                entity_name="document element",
            )

        example_id = normalized.example.example_id

        if example_id in examples_by_id:
            raise ValueError(f"Duplicate dataset example ID: {example_id!r}")

        examples_by_id[example_id] = normalized.example

    return PreparedFinQASplit(
        split=split,
        documents=tuple(
            sorted(
                documents_by_id.values(),
                key=lambda document: document.document_id,
            )
        ),
        elements=tuple(
            sorted(
                elements_by_id.values(),
                key=lambda element: element.element_id,
            )
        ),
        examples=tuple(
            sorted(
                examples_by_id.values(),
                key=lambda example: example.example_id,
            )
        ),
    )


def _store_reusable_entity[T](
    entities: dict[str, T],
    *,
    entity_id: str,
    entity: T,
    entity_name: str,
) -> None:
    existing = entities.get(entity_id)

    if existing is not None and existing != entity:
        raise ValueError(f"Conflicting {entity_name} with ID {entity_id!r}")

    entities[entity_id] = entity
