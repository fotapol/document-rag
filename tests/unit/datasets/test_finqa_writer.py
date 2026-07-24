import json
from pathlib import Path

from document_rag.datasets import (
    DatasetExample,
    DatasetName,
    DatasetSplit,
    ReferenceAnswer,
    SupportingFact,
)
from document_rag.datasets.finqa import (
    PreparedFinQASplit,
    write_finqa_split,
)
from document_rag.domain import (
    Document,
    DocumentElement,
    DocumentElementType,
    Question,
)


def make_prepared_split() -> PreparedFinQASplit:
    document = Document(
        document_id="finqa:ABC/2025/page_10.pdf",
        file_name="page_10.pdf",
        source_uri="ABC/2025/page_10.pdf",
        page_count=1,
    )

    element = DocumentElement(
        element_id="finqa:ABC/2025/page_10.pdf:text_0",
        document_id=document.document_id,
        element_type=DocumentElementType.PARAGRAPH,
        source_text="Revenue increased during the year.",
        page_number=10,
    )

    question = Question(
        question_id="finqa:ABC/2025/page_10.pdf-1",
        document_id=document.document_id,
        text="Did revenue increase?",
    )

    example = DatasetExample(
        dataset=DatasetName.FINQA,
        split=DatasetSplit.TRAIN,
        example_id=question.question_id,
        question=question,
        reference_answer=ReferenceAnswer(text="yes"),
        supporting_facts=(
            SupportingFact(
                source_key="text_0",
                element_id=element.element_id,
            ),
        ),
    )

    return PreparedFinQASplit(
        split=DatasetSplit.TRAIN,
        documents=(document,),
        elements=(element,),
        examples=(example,),
    )


def test_writer_creates_normalized_jsonl_files(tmp_path: Path) -> None:
    result = write_finqa_split(
        make_prepared_split(),
        output_directory=tmp_path,
    )

    assert result.documents.path.is_file()
    assert result.elements.path.is_file()
    assert result.examples.path.is_file()

    document_payload = json.loads(result.documents.path.read_text(encoding="utf-8").strip())

    assert document_payload["document_id"] == "finqa:ABC/2025/page_10.pdf"
    assert result.documents.record_count == 1


def test_writer_produces_identical_checksums(tmp_path: Path) -> None:
    prepared_split = make_prepared_split()

    first = write_finqa_split(
        prepared_split,
        output_directory=tmp_path,
    )
    second = write_finqa_split(
        prepared_split,
        output_directory=tmp_path,
    )

    assert first.documents.checksum_sha256 == second.documents.checksum_sha256
    assert first.elements.checksum_sha256 == second.elements.checksum_sha256
    assert first.examples.checksum_sha256 == second.examples.checksum_sha256


def test_writer_leaves_no_temporary_files(tmp_path: Path) -> None:
    write_finqa_split(
        make_prepared_split(),
        output_directory=tmp_path,
    )

    temporary_files = list((tmp_path / "train").glob("*.tmp"))

    assert temporary_files == []
