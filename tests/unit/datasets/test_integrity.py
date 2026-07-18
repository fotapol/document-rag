from dataclasses import replace

import pytest

from document_rag.datasets import (
    DatasetExample,
    DatasetName,
    DatasetSplit,
    ReferenceAnswer,
    SupportingFact,
)
from document_rag.datasets.finqa import PreparedFinQASplit
from document_rag.datasets.integrity import (
    build_split_integrity_snapshot,
    validate_cross_split_integrity,
)
from document_rag.domain import (
    Document,
    DocumentElement,
    DocumentElementType,
    Question,
)


def make_prepared_split(
    *,
    split: DatasetSplit = DatasetSplit.TRAIN,
    document_id: str = "finqa:ABC/2025/page_10.pdf",
    source_uri: str = "ABC/2025/page_10.pdf",
    example_id: str = "finqa:ABC/2025/page_10.pdf-1",
) -> PreparedFinQASplit:
    document = Document(
        document_id=document_id,
        file_name=source_uri.rsplit("/", maxsplit=1)[-1],
        source_uri=source_uri,
        page_count=1,
    )

    element = DocumentElement(
        element_id=f"{document_id}:text_0",
        document_id=document_id,
        element_type=DocumentElementType.PARAGRAPH,
        source_text="Revenue increased.",
        page_number=10,
    )

    question = Question(
        question_id=example_id,
        document_id=document_id,
        text="Did revenue increase?",
    )

    example = DatasetExample(
        dataset=DatasetName.FINQA,
        split=split,
        example_id=example_id,
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
        split=split,
        documents=(document,),
        elements=(element,),
        examples=(example,),
    )


def test_snapshot_collects_split_identifiers() -> None:
    prepared_split = make_prepared_split()

    snapshot = build_split_integrity_snapshot(prepared_split)

    assert snapshot.split is DatasetSplit.TRAIN
    assert snapshot.example_ids == frozenset({"finqa:ABC/2025/page_10.pdf-1"})
    assert snapshot.document_ids == frozenset({"finqa:ABC/2025/page_10.pdf"})
    assert snapshot.report_ids == frozenset({"ABC/2025"})


def test_snapshot_rejects_element_with_unknown_document() -> None:
    prepared_split = make_prepared_split()

    invalid_element = prepared_split.elements[0].model_copy(
        update={"document_id": "finqa:missing.pdf"}
    )
    invalid_split = replace(
        prepared_split,
        elements=(invalid_element,),
    )

    with pytest.raises(ValueError, match="references unknown document"):
        build_split_integrity_snapshot(invalid_split)


def test_snapshot_rejects_unknown_supporting_element() -> None:
    prepared_split = make_prepared_split()

    invalid_example = prepared_split.examples[0].model_copy(
        update={
            "supporting_facts": (
                SupportingFact(
                    source_key="text_999",
                    element_id="finqa:missing:text_999",
                ),
            )
        }
    )
    invalid_split = replace(
        prepared_split,
        examples=(invalid_example,),
    )

    with pytest.raises(ValueError, match="references unknown element"):
        build_split_integrity_snapshot(invalid_split)


def test_snapshot_rejects_duplicate_example_ids() -> None:
    prepared_split = make_prepared_split()
    duplicate_split = replace(
        prepared_split,
        examples=(
            prepared_split.examples[0],
            prepared_split.examples[0],
        ),
    )

    with pytest.raises(ValueError, match="duplicate example IDs"):
        build_split_integrity_snapshot(duplicate_split)


def test_cross_split_validation_rejects_shared_examples() -> None:
    train_snapshot = build_split_integrity_snapshot(make_prepared_split())
    validation_snapshot = build_split_integrity_snapshot(
        make_prepared_split(
            split=DatasetSplit.VALIDATION,
            document_id="finqa:XYZ/2024/page_20.pdf",
            source_uri="XYZ/2024/page_20.pdf",
            example_id="finqa:ABC/2025/page_10.pdf-1",
        )
    )

    with pytest.raises(ValueError, match="share 1 example IDs"):
        validate_cross_split_integrity(
            validation_snapshot,
            previous_snapshots=(train_snapshot,),
        )


def test_cross_split_validation_rejects_shared_documents() -> None:
    train_snapshot = build_split_integrity_snapshot(make_prepared_split())
    validation_snapshot = build_split_integrity_snapshot(
        make_prepared_split(
            split=DatasetSplit.VALIDATION,
            example_id="finqa:ABC/2025/page_10.pdf-2",
        )
    )

    with pytest.raises(ValueError, match="share 1 documents"):
        validate_cross_split_integrity(
            validation_snapshot,
            previous_snapshots=(train_snapshot,),
        )


def test_cross_split_validation_reports_shared_reports() -> None:
    train_snapshot = build_split_integrity_snapshot(make_prepared_split())
    validation_snapshot = build_split_integrity_snapshot(
        make_prepared_split(
            split=DatasetSplit.VALIDATION,
            document_id="finqa:ABC/2025/page_11.pdf",
            source_uri="ABC/2025/page_11.pdf",
            example_id="finqa:ABC/2025/page_11.pdf-1",
        )
    )

    overlaps = validate_cross_split_integrity(
        validation_snapshot,
        previous_snapshots=(train_snapshot,),
    )

    assert len(overlaps) == 1
    assert overlaps[0].count == 1
    assert overlaps[0].report_ids == ("ABC/2025",)
    assert overlaps[0].left_split is DatasetSplit.TRAIN
    assert overlaps[0].right_split is DatasetSplit.VALIDATION
