from decimal import Decimal

import pytest
from pydantic import ValidationError

from document_rag.domain import (
    Answer,
    AnswerStatus,
    Calculation,
    CalculationOperand,
    CalculationOperation,
    Citation,
)


def make_citation() -> Citation:
    return Citation(
        citation_id="citation-1",
        document_id="report-2025",
        evidence_ids=("evidence-1",),
        source_element_ids=("table-row-1",),
        page_numbers=(42,),
        excerpt="Revenue increased from 100 to 120 million.",
    )


def make_calculation() -> Calculation:
    return Calculation(
        calculation_id="calculation-1",
        question_id="question-1",
        operation=CalculationOperation.PERCENTAGE_CHANGE,
        operands=(
            CalculationOperand(
                label="Revenue 2024",
                value=Decimal("100"),
                source_evidence_ids=("evidence-1",),
            ),
            CalculationOperand(
                label="Revenue 2025",
                value=Decimal("120"),
                source_evidence_ids=("evidence-1",),
            ),
        ),
        result=Decimal("20"),
        result_unit="percent",
        display_formula="(120 - 100) / 100 x 100",
    )


def test_answer_accepts_grounded_response() -> None:
    answer = Answer(
        answer_id="answer-1",
        question_id="question-1",
        text="Revenue increased by 20 percent.",
        status=AnswerStatus.ANSWERED,
        citations=(make_citation(),),
        calculations=(make_calculation(),),
    )

    assert answer.status is AnswerStatus.ANSWERED
    assert answer.calculations[0].result == Decimal("20")


def test_answered_response_requires_citation() -> None:
    with pytest.raises(
        ValidationError,
        match="requires at least one citation",
    ):
        Answer(
            answer_id="answer-1",
            question_id="question-1",
            text="Revenue increased by 20 percent.",
            status=AnswerStatus.ANSWERED,
        )


def test_insufficient_evidence_answer_can_have_no_citations() -> None:
    answer = Answer(
        answer_id="answer-1",
        question_id="question-1",
        text="The document does not contain enough information.",
        status=AnswerStatus.INSUFFICIENT_EVIDENCE,
    )

    assert answer.citations == ()
    assert answer.calculations == ()


def test_answer_rejects_calculation_for_another_question() -> None:
    calculation = make_calculation().model_copy(update={"question_id": "question-2"})

    with pytest.raises(
        ValidationError,
        match="must belong to the answer question",
    ):
        Answer(
            answer_id="answer-1",
            question_id="question-1",
            text="Revenue increased by 20 percent.",
            status=AnswerStatus.ANSWERED,
            citations=(make_citation(),),
            calculations=(calculation,),
        )


def test_answer_rejects_duplicate_citation_ids() -> None:
    citation = make_citation()

    with pytest.raises(
        ValidationError,
        match="Citation IDs must be unique",
    ):
        Answer(
            answer_id="answer-1",
            question_id="question-1",
            text="Revenue increased.",
            status=AnswerStatus.ANSWERED,
            citations=(citation, citation),
        )


def test_answer_serializes_nested_models() -> None:
    answer = Answer(
        answer_id="answer-1",
        question_id="question-1",
        text="Revenue increased by 20 percent.",
        status=AnswerStatus.ANSWERED,
        citations=(make_citation(),),
        calculations=(make_calculation(),),
    )

    result = answer.model_dump(mode="json")

    assert result["status"] == "answered"
    assert result["citations"][0]["page_numbers"] == [42]
    assert result["calculations"][0]["result"] == "20"
