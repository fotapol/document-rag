import json
from pathlib import Path

import pytest

from document_rag.datasets.docfinqa import (
    DocFinQALinkStatus,
    NormalizedDocFinQAElement,
    NormalizedDocFinQASupportingFact,
    PreparedDocFinQADocument,
    PreparedDocFinQAExample,
    PreparedDocFinQAItem,
)
from document_rag.datasets.docfinqa.writer import (
    write_docfinqa_split,
)
from document_rag.datasets.models import (
    DatasetName,
    DatasetSplit,
)


def make_item(
    *,
    document_id: str = "docfinqa:document:abc",
    element_id: str = ("docfinqa:document:abc:char-2750-550:chunk-000000"),
    example_id: str = "docfinqa:example:abc",
    include_document: bool = True,
) -> PreparedDocFinQAItem:
    document = (
        PreparedDocFinQADocument(
            dataset=DatasetName.DOCFINQA,
            split=DatasetSplit.TRAIN,
            document_id=document_id,
        )
        if include_document
        else None
    )

    elements = (
        (
            NormalizedDocFinQAElement(
                element_id=element_id,
                document_id=document_id,
                index=0,
                start_char=0,
                end_char=19,
                source_text="Revenue was 100.",
            ),
        )
        if include_document
        else ()
    )

    example = PreparedDocFinQAExample(
        dataset=DatasetName.DOCFINQA,
        split=DatasetSplit.TRAIN,
        document_id=document_id,
        example_id=example_id,
        finqa_id="ABC/2020/page_1.pdf-1",
        finqa_source_file="ABC/2020/page_1.pdf",
        link_status=DocFinQALinkStatus.EXACT,
        question="What was the revenue?",
        answer="100",
        program="answer = 100",
        supporting_facts=(
            NormalizedDocFinQASupportingFact(
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


def read_jsonl(path: Path) -> list[dict[str, object]]:
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

    assert documents == [
        {
            "dataset": "docfinqa",
            "document_id": ("docfinqa:document:abc"),
            "split": "train",
        }
    ]

    assert elements[0]["source_text"] == ("Revenue was 100.")
    assert examples[0]["question"] == ("What was the revenue?")
    assert examples[0]["link_status"] == "exact"


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

    invalid_example = PreparedDocFinQAExample(
        dataset=item.example.dataset,
        split=item.example.split,
        document_id=item.example.document_id,
        example_id=item.example.example_id,
        finqa_id=item.example.finqa_id,
        finqa_source_file=(item.example.finqa_source_file),
        link_status=item.example.link_status,
        question=item.example.question,
        answer=item.example.answer,
        program=item.example.program,
        supporting_facts=(
            NormalizedDocFinQASupportingFact(
                source_key="text_1",
                element_id="unknown-element",
                score=1.0,
            ),
        ),
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
