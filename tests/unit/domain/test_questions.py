import pytest
from pydantic import ValidationError

from document_rag.domain import Question, QuestionType


def test_question_uses_unknown_type_by_default() -> None:
    question = Question(
        question_id="question-1",
        document_id="report-2025",
        text="What was the annual revenue?",
    )

    assert question.question_type is QuestionType.UNKNOWN


def test_question_accepts_explicit_type() -> None:
    question = Question(
        question_id="question-1",
        document_id="report-2025",
        text="How much did revenue increase?",
        question_type=QuestionType.CALCULATION,
    )

    assert question.question_type is QuestionType.CALCULATION


def test_question_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        Question(
            question_id="question-1",
            document_id="report-2025",
            text="",
        )


def test_question_serializes_enum_as_string() -> None:
    question = Question(
        question_id="question-1",
        document_id="report-2025",
        text="Compare revenue between the two years.",
        question_type=QuestionType.COMPARISON,
    )

    result = question.model_dump(mode="json")

    assert result["question_type"] == "comparison"
