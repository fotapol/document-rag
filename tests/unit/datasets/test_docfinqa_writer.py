import json
from pathlib import Path
from typing import Any

import pytest

from document_rag.datasets.docfinqa import (
    PreparedDocFinQAItem,
)
from document_rag.datasets.docfinqa.writer import (
    write_docfinqa_split,
)
from document_rag.datasets.models import (
    DatasetExample,
    DatasetName,
    DatasetSplit,
    ReferenceAnswer,
    SupportingFact,
)
from document_rag.domain import (
    Document,
    DocumentElement,
    DocumentElementType,
    Question,
)


def make_item(
    *,
    document_id: str = "docfinqa:document:abc",
    element_id: str = ("docfinqa:document:abc:char-2750-550:chunk-000000"),
    example_id: str = "docfinqa:example:abc",
    include_document: bool = True,
) -> PreparedDocFinQAItem:
    document = (
        Document(
            document_id=document_id,
            file_name="abc.txt",
            mime_type="text/plain",
            page_count=1,
            metadata={
                "dataset": DatasetName.DOCFINQA.value,
                "split": DatasetSplit.TRAIN.value,
            },
        )
        if include_document
        else None
    )

    elements = (
        (
            DocumentElement(
                element_id=element_id,
                document_id=document_id,
                element_type=DocumentElementType.PARAGRAPH,
                source_text="Revenue was 100.",
                page_number=1,
                metadata={
                    "dataset": DatasetName.DOCFINQA.value,
                    "split": DatasetSplit.TRAIN.value,
                    "chunk_index": 0,
                    "start_char": 0,
                    "end_char": 16,
                },
            ),
        )
        if include_document
        else ()
    )

    example = DatasetExample(
        dataset=DatasetName.DOCFINQA,
        split=DatasetSplit.TRAIN,
        example_id=example_id,
        question=Question(
            question_id=example_id,
            document_id=document_id,
            text="What was the revenue?",
            metadata={
                "source_example_id": "ABC/2020/page_1.pdf-1",
                "source_file": "ABC/2020/page_1.pdf",
                "link_status": "exact",
            },
        ),
        reference_answer=ReferenceAnswer(
            text="100",
            program="answer = 100",
        ),
        supporting_facts=(
            SupportingFact(
                source_key="text_1",
                element_id=element_id,
                score=1.0,
            ),
        ),
    )

    return PreparedDocFinQAItem(
        document=document,
        elements=elements,
        example=example,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_writer_creates_split_artifacts(
    tmp_path: Path,
) -> None:
    result = write_docfinqa_split(
        output_directory=tmp_path,
        split=DatasetSplit.TRAIN,
        items=[make_item()],
    )

    assert result.split is DatasetSplit.TRAIN

    assert result.documents.record_count == 1
    assert result.elements.record_count == 1
    assert result.examples.record_count == 1

    assert len(result.documents.sha256) == 64
    assert len(result.elements.sha256) == 64
    assert len(result.examples.sha256) == 64

    documents = read_jsonl(result.documents.path)
    elements = read_jsonl(result.elements.path)
    examples = read_jsonl(result.examples.path)

    assert documents[0]["document_id"] == "docfinqa:document:abc"
    assert documents[0]["metadata"] == {
        "dataset": "docfinqa",
        "split": "train",
    }

    assert elements[0]["source_text"] == ("Revenue was 100.")
    assert examples[0]["question"]["text"] == ("What was the revenue?")
    assert examples[0]["question"]["question_id"] == ("docfinqa:example:abc")
    assert examples[0]["question"]["metadata"]["link_status"] == "exact"


def test_writer_supports_multiple_questions_per_document(
    tmp_path: Path,
) -> None:
    first_item = make_item()

    second_item = make_item(
        example_id="docfinqa:example:def",
        include_document=False,
    )

    result = write_docfinqa_split(
        output_directory=tmp_path,
        split=DatasetSplit.TRAIN,
        items=[
            first_item,
            second_item,
        ],
    )

    assert result.documents.record_count == 1
    assert result.elements.record_count == 1
    assert result.examples.record_count == 2


def test_writer_is_deterministic(
    tmp_path: Path,
) -> None:
    first_result = write_docfinqa_split(
        output_directory=tmp_path / "first",
        split=DatasetSplit.TRAIN,
        items=[make_item()],
    )

    second_result = write_docfinqa_split(
        output_directory=tmp_path / "second",
        split=DatasetSplit.TRAIN,
        items=[make_item()],
    )

    artifact_pairs = [
        (
            first_result.documents,
            second_result.documents,
        ),
        (
            first_result.elements,
            second_result.elements,
        ),
        (
            first_result.examples,
            second_result.examples,
        ),
    ]

    for first_artifact, second_artifact in artifact_pairs:
        assert first_artifact.sha256 == second_artifact.sha256
        assert first_artifact.path.read_bytes() == second_artifact.path.read_bytes()


def test_writer_rejects_unknown_document(
    tmp_path: Path,
) -> None:
    item = make_item(
        include_document=False,
    )

    with pytest.raises(
        ValueError,
        match="unknown document",
    ):
        write_docfinqa_split(
            output_directory=tmp_path,
            split=DatasetSplit.TRAIN,
            items=[item],
        )


def test_writer_rejects_unknown_supporting_element(
    tmp_path: Path,
) -> None:
    item = make_item()

    invalid_example = item.example.model_copy(
        update={
            "supporting_facts": (
                SupportingFact(
                    source_key="text_1",
                    element_id="unknown-element",
                    score=1.0,
                ),
            )
        }
    )

    invalid_item = PreparedDocFinQAItem(
        document=item.document,
        elements=item.elements,
        example=invalid_example,
    )

    with pytest.raises(
        ValueError,
        match="unknown element",
    ):
        write_docfinqa_split(
            output_directory=tmp_path,
            split=DatasetSplit.TRAIN,
            items=[invalid_item],
        )


def test_writer_rejects_duplicate_example_id(
    tmp_path: Path,
) -> None:
    first_item = make_item()
    second_item = make_item(
        include_document=False,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate DocFinQA example ID",
    ):
        write_docfinqa_split(
            output_directory=tmp_path,
            split=DatasetSplit.TRAIN,
            items=[
                first_item,
                second_item,
            ],
        )


def test_writer_rejects_empty_stream(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="Prepared DocFinQA split is empty",
    ):
        write_docfinqa_split(
            output_directory=tmp_path,
            split=DatasetSplit.TRAIN,
            items=[],
        )


def test_failed_write_preserves_existing_artifacts(
    tmp_path: Path,
) -> None:
    first_result = write_docfinqa_split(
        output_directory=tmp_path,
        split=DatasetSplit.TRAIN,
        items=[make_item()],
    )

    original_documents = first_result.documents.path.read_bytes()
    original_elements = first_result.elements.path.read_bytes()
    original_examples = first_result.examples.path.read_bytes()

    with pytest.raises(
        ValueError,
        match="unknown document",
    ):
        write_docfinqa_split(
            output_directory=tmp_path,
            split=DatasetSplit.TRAIN,
            items=[
                make_item(
                    include_document=False,
                )
            ],
        )

    assert first_result.documents.path.read_bytes() == original_documents
    assert first_result.elements.path.read_bytes() == original_elements
    assert first_result.examples.path.read_bytes() == original_examples
