import pytest
from pydantic import ValidationError

from document_rag.datasets import (
    DatasetExample,
    DatasetName,
    DatasetSplit,
    ReasoningStep,
    ReferenceAnswer,
    SupportingFact,
)
from document_rag.domain import Question


def test_dataset_example_accepts_normalized_annotations() -> None:
    example = DatasetExample(
        dataset=DatasetName.FINQA,
        split=DatasetSplit.TRAIN,
        example_id="ADI/2009/page_49.pdf-1",
        question=Question(
            question_id="ADI/2009/page_49.pdf-1",
            document_id="ADI/2009/page_49.pdf",
            text="What is the interest expense in 2009?",
        ),
        reference_answer=ReferenceAnswer(
            text="380",
            executable_answer=3.8,
            program="divide(100, 100), divide(3.8, #0)",
            steps=(
                ReasoningStep(
                    operation="divide1-1",
                    arguments=("100", "100"),
                    result="1%",
                ),
            ),
        ),
        supporting_facts=(
            SupportingFact(
                source_key="text_1",
                element_id="ADI/2009/page_49.pdf:text:1",
            ),
        ),
    )

    assert example.dataset is DatasetName.FINQA
    assert example.split is DatasetSplit.TRAIN
    assert example.supporting_facts[0].source_key == "text_1"


def test_dataset_example_requires_supporting_facts() -> None:
    with pytest.raises(ValidationError):
        DatasetExample(
            dataset=DatasetName.FINQA,
            split=DatasetSplit.TRAIN,
            example_id="example-1",
            question=Question(
                question_id="question-1",
                document_id="document-1",
                text="What was the revenue?",
            ),
            reference_answer=ReferenceAnswer(text="100"),
            supporting_facts=(),
        )


def test_reference_answer_serializes_reasoning_steps() -> None:
    answer = ReferenceAnswer(
        text="20%",
        steps=(
            ReasoningStep(
                operation="subtract",
                arguments=("120", "100"),
                result="20",
            ),
        ),
    )

    result = answer.model_dump(mode="json")

    assert result["steps"][0]["operation"] == "subtract"
    assert result["steps"][0]["arguments"] == ["120", "100"]


def test_dataset_split_uses_validation_name() -> None:
    assert DatasetSplit.VALIDATION.value == "validation"
