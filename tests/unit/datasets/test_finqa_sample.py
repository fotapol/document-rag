import pytest

from document_rag.datasets import (
    DatasetExample,
    DatasetName,
    DatasetSplit,
    ReferenceAnswer,
    SupportingFact,
)
from document_rag.datasets.finqa.preparer import PreparedFinQASplit
from document_rag.datasets.finqa.sample import create_finqa_sample
from document_rag.domain import (
    Document,
    DocumentElement,
    DocumentElementType,
    Question,
)


def make_prepared_split(
    *,
    reverse_order: bool = False,
) -> PreparedFinQASplit:
    documents: list[Document] = []
    elements: list[DocumentElement] = []
    examples: list[DatasetExample] = []

    for index in range(1, 5):
        document_id = f"finqa:ABC/2025/page_{index}.pdf"
        example_id = f"{document_id}-1"

        document = Document(
            document_id=document_id,
            file_name=f"page_{index}.pdf",
            source_uri=f"ABC/2025/page_{index}.pdf",
            page_count=1,
        )

        evidence_element = DocumentElement(
            element_id=f"{document_id}:text_0",
            document_id=document_id,
            element_type=DocumentElementType.PARAGRAPH,
            source_text=f"Revenue evidence for document {index}.",
            page_number=index,
        )

        negative_element = DocumentElement(
            element_id=f"{document_id}:text_1",
            document_id=document_id,
            element_type=DocumentElementType.PARAGRAPH,
            source_text=f"Unrelated paragraph for document {index}.",
            page_number=index,
        )

        question = Question(
            question_id=example_id,
            document_id=document_id,
            text=f"What was the revenue for document {index}?",
        )

        example = DatasetExample(
            dataset=DatasetName.FINQA,
            split=DatasetSplit.TRAIN,
            example_id=example_id,
            question=question,
            reference_answer=ReferenceAnswer(text=str(index * 100)),
            supporting_facts=(
                SupportingFact(
                    source_key="text_0",
                    element_id=evidence_element.element_id,
                ),
            ),
        )

        documents.append(document)
        elements.extend((evidence_element, negative_element))
        examples.append(example)

    if reverse_order:
        documents.reverse()
        elements.reverse()
        examples.reverse()

    return PreparedFinQASplit(
        split=DatasetSplit.TRAIN,
        documents=tuple(documents),
        elements=tuple(elements),
        examples=tuple(examples),
    )


def test_sample_selects_requested_number_of_examples() -> None:
    result = create_finqa_sample(
        make_prepared_split(),
        example_count=2,
    )

    assert len(result.examples) == 2
    assert result.split is DatasetSplit.TRAIN


def test_sample_includes_complete_selected_documents() -> None:
    result = create_finqa_sample(
        make_prepared_split(),
        example_count=1,
    )

    assert len(result.documents) == 1

    selected_document_id = result.examples[0].question.document_id
    selected_elements = [
        element for element in result.elements if element.document_id == selected_document_id
    ]

    assert len(selected_elements) == 2
    assert {element.element_type for element in selected_elements} == {
        DocumentElementType.PARAGRAPH
    }


def test_sample_is_independent_of_source_order() -> None:
    first = create_finqa_sample(
        make_prepared_split(),
        example_count=2,
    )
    second = create_finqa_sample(
        make_prepared_split(reverse_order=True),
        example_count=2,
    )

    first_ids = tuple(example.example_id for example in first.examples)
    second_ids = tuple(example.example_id for example in second.examples)

    assert first_ids == second_ids
    assert first == second


def test_sample_changes_when_seed_changes() -> None:
    first = create_finqa_sample(
        make_prepared_split(),
        example_count=2,
        seed="first-seed",
    )
    second = create_finqa_sample(
        make_prepared_split(),
        example_count=2,
        seed="second-seed",
    )

    first_ids = {example.example_id for example in first.examples}
    second_ids = {example.example_id for example in second.examples}

    assert first_ids != second_ids


@pytest.mark.parametrize(
    "example_count",
    [0, -1],
)
def test_sample_rejects_non_positive_size(
    example_count: int,
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        create_finqa_sample(
            make_prepared_split(),
            example_count=example_count,
        )


def test_sample_rejects_size_larger_than_split() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        create_finqa_sample(
            make_prepared_split(),
            example_count=5,
        )


def test_sample_rejects_empty_seed() -> None:
    with pytest.raises(ValueError, match="seed cannot be empty"):
        create_finqa_sample(
            make_prepared_split(),
            example_count=1,
            seed=" ",
        )
