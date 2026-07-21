"""Deterministic streaming JSONL writer for prepared DocFinQA splits."""

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import IO, Any

from document_rag.datasets.docfinqa.normalizer import (
    NormalizedDocFinQAElement,
    NormalizedDocFinQASupportingFact,
)
from document_rag.datasets.docfinqa.preparer import (
    PreparedDocFinQADocument,
    PreparedDocFinQAExample,
    PreparedDocFinQAItem,
)
from document_rag.datasets.models import (
    DatasetName,
    DatasetSplit,
)


@dataclass(frozen=True, slots=True)
class WrittenDocFinQAArtifact:
    """Metadata for one written JSONL artifact."""

    path: Path
    record_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class WrittenDocFinQASplit:
    """Artifacts produced for one DocFinQA split."""

    split: DatasetSplit
    documents: WrittenDocFinQAArtifact
    elements: WrittenDocFinQAArtifact
    examples: WrittenDocFinQAArtifact


class _TemporaryJsonlWriter:
    """Write deterministic JSONL before atomically replacing a target."""

    def __init__(
        self,
        *,
        target_path: Path,
        temporary_path: Path,
        temporary_file: IO[bytes],
    ) -> None:
        self._target_path = target_path
        self._temporary_path = temporary_path
        self._file = temporary_file
        self._digest = sha256()
        self._record_count = 0
        self._closed = False
        self._committed = False

    @property
    def record_count(self) -> int:
        return self._record_count

    @property
    def checksum(self) -> str:
        return self._digest.hexdigest()

    def write(self, payload: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("Cannot write to a closed JSONL writer")

        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

        line = serialized + b"\n"

        self._file.write(line)
        self._digest.update(line)
        self._record_count += 1

    def commit(self) -> WrittenDocFinQAArtifact:
        self._close()
        os.replace(
            self._temporary_path,
            self._target_path,
        )
        self._committed = True

        return WrittenDocFinQAArtifact(
            path=self._target_path,
            record_count=self._record_count,
            sha256=self.checksum,
        )

    def abort(self) -> None:
        self._close()

        if not self._committed:
            self._temporary_path.unlink(
                missing_ok=True,
            )

    def _close(self) -> None:
        if self._closed:
            return

        self._file.close()
        self._closed = True


@contextmanager
def _open_temporary_jsonl_writer(
    target_path: Path,
) -> Iterator[_TemporaryJsonlWriter]:
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{target_path.name}.",
        suffix=".tmp",
        dir=target_path.parent,
        delete=False,
    ) as temporary_file:
        writer = _TemporaryJsonlWriter(
            target_path=target_path,
            temporary_path=Path(temporary_file.name),
            temporary_file=temporary_file.file,
        )

        try:
            yield writer
        finally:
            writer.abort()


def write_docfinqa_split(
    *,
    output_directory: Path,
    split: DatasetSplit,
    items: Iterable[PreparedDocFinQAItem],
) -> WrittenDocFinQASplit:
    """Write one prepared split using bounded memory.

    The function verifies that:

    - documents, elements and examples have unique IDs;
    - every example references an emitted document;
    - every supporting fact references an emitted element;
    - dataset and split values match the current writer invocation.

    Existing artifact files are replaced only after the full stream
    has been validated and serialized successfully.
    """

    split_directory = output_directory / split.value
    split_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        _open_temporary_jsonl_writer(split_directory / "documents.jsonl") as document_writer,
        _open_temporary_jsonl_writer(split_directory / "elements.jsonl") as element_writer,
        _open_temporary_jsonl_writer(split_directory / "examples.jsonl") as example_writer,
    ):
        known_document_ids: set[str] = set()
        known_element_ids: set[str] = set()
        known_example_ids: set[str] = set()

        for item in items:
            _validate_and_write_item(
                item=item,
                split=split,
                known_document_ids=known_document_ids,
                known_element_ids=known_element_ids,
                known_example_ids=known_example_ids,
                document_writer=document_writer,
                element_writer=element_writer,
                example_writer=example_writer,
            )

        if example_writer.record_count == 0:
            raise ValueError(f"Prepared DocFinQA split is empty: {split.value}")

        documents = document_writer.commit()
        elements = element_writer.commit()
        examples = example_writer.commit()

    return WrittenDocFinQASplit(
        split=split,
        documents=documents,
        elements=elements,
        examples=examples,
    )


def _validate_and_write_item(
    *,
    item: PreparedDocFinQAItem,
    split: DatasetSplit,
    known_document_ids: set[str],
    known_element_ids: set[str],
    known_example_ids: set[str],
    document_writer: _TemporaryJsonlWriter,
    element_writer: _TemporaryJsonlWriter,
    example_writer: _TemporaryJsonlWriter,
) -> None:
    example = item.example

    _validate_example_identity(
        example=example,
        split=split,
    )

    if item.document is not None:
        _write_new_document(
            document=item.document,
            elements=item.elements,
            example=example,
            split=split,
            known_document_ids=known_document_ids,
            known_element_ids=known_element_ids,
            document_writer=document_writer,
            element_writer=element_writer,
        )
    else:
        if item.elements:
            raise ValueError("Elements cannot be emitted without a document")

        if example.document_id not in known_document_ids:
            raise ValueError(f"Example references an unknown document: {example.document_id}")

    if example.example_id in known_example_ids:
        raise ValueError(f"Duplicate DocFinQA example ID: {example.example_id}")

    for supporting_fact in example.supporting_facts:
        if supporting_fact.element_id not in known_element_ids:
            raise ValueError(
                f"Supporting fact references an unknown element: {supporting_fact.element_id}"
            )

    known_example_ids.add(example.example_id)
    example_writer.write(_serialize_example(example))


def _write_new_document(
    *,
    document: PreparedDocFinQADocument,
    elements: tuple[NormalizedDocFinQAElement, ...],
    example: PreparedDocFinQAExample,
    split: DatasetSplit,
    known_document_ids: set[str],
    known_element_ids: set[str],
    document_writer: _TemporaryJsonlWriter,
    element_writer: _TemporaryJsonlWriter,
) -> None:
    if document.dataset is not DatasetName.DOCFINQA:
        raise ValueError("Document dataset must be docfinqa")

    if document.split is not split:
        raise ValueError("Document split does not match writer split")

    if document.document_id != example.document_id:
        raise ValueError("Document and example IDs do not match")

    if document.document_id in known_document_ids:
        raise ValueError(f"Duplicate DocFinQA document ID: {document.document_id}")

    if not elements:
        raise ValueError("A new document must contain at least one element")

    known_document_ids.add(document.document_id)
    document_writer.write(_serialize_document(document))

    for element in elements:
        if element.document_id != document.document_id:
            raise ValueError(f"Element references the wrong document: {element.element_id}")

        if element.element_id in known_element_ids:
            raise ValueError(f"Duplicate DocFinQA element ID: {element.element_id}")

        known_element_ids.add(element.element_id)
        element_writer.write(_serialize_element(element))


def _validate_example_identity(
    *,
    example: PreparedDocFinQAExample,
    split: DatasetSplit,
) -> None:
    if example.dataset is not DatasetName.DOCFINQA:
        raise ValueError("Example dataset must be docfinqa")

    if example.split is not split:
        raise ValueError("Example split does not match writer split")


def _serialize_document(
    document: PreparedDocFinQADocument,
) -> dict[str, Any]:
    return {
        "dataset": document.dataset.value,
        "document_id": document.document_id,
        "split": document.split.value,
    }


def _serialize_element(
    element: NormalizedDocFinQAElement,
) -> dict[str, Any]:
    return {
        "document_id": element.document_id,
        "element_id": element.element_id,
        "end_char": element.end_char,
        "index": element.index,
        "source_text": element.source_text,
        "start_char": element.start_char,
    }


def _serialize_example(
    example: PreparedDocFinQAExample,
) -> dict[str, Any]:
    return {
        "answer": example.answer,
        "dataset": example.dataset.value,
        "document_id": example.document_id,
        "example_id": example.example_id,
        "finqa_id": example.finqa_id,
        "finqa_source_file": example.finqa_source_file,
        "link_status": example.link_status.value,
        "program": example.program,
        "question": example.question,
        "split": example.split.value,
        "supporting_facts": [_serialize_supporting_fact(fact) for fact in example.supporting_facts],
    }


def _serialize_supporting_fact(
    supporting_fact: NormalizedDocFinQASupportingFact,
) -> dict[str, Any]:
    return {
        "element_id": supporting_fact.element_id,
        "score": supporting_fact.score,
        "source_key": supporting_fact.source_key,
    }
