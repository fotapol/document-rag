import pytest
from pydantic import ValidationError

from document_rag.domain import Evidence


def test_evidence_accepts_valid_retrieval_result() -> None:
    evidence = Evidence(
        evidence_id="evidence-1",
        question_id="question-1",
        document_id="report-2025",
        text="Revenue increased from 100 to 120 million dollars.",
        source_element_ids=("paragraph-10", "table-row-4"),
        page_numbers=(14, 15),
        rank=1,
        retrieval_score=8.42,
        reranker_score=3.17,
        retriever="hybrid-rrf",
    )

    assert evidence.rank == 1
    assert evidence.source_element_ids == ("paragraph-10", "table-row-4")
    assert evidence.page_numbers == (14, 15)


def test_evidence_requires_source_elements() -> None:
    with pytest.raises(ValidationError):
        Evidence(
            evidence_id="evidence-1",
            question_id="question-1",
            document_id="report-2025",
            text="Revenue increased.",
            source_element_ids=(),
            page_numbers=(14,),
            rank=1,
        )


def test_evidence_requires_page_numbers() -> None:
    with pytest.raises(ValidationError):
        Evidence(
            evidence_id="evidence-1",
            question_id="question-1",
            document_id="report-2025",
            text="Revenue increased.",
            source_element_ids=("paragraph-10",),
            page_numbers=(),
            rank=1,
        )


def test_evidence_rejects_zero_rank() -> None:
    with pytest.raises(ValidationError):
        Evidence(
            evidence_id="evidence-1",
            question_id="question-1",
            document_id="report-2025",
            text="Revenue increased.",
            source_element_ids=("paragraph-10",),
            page_numbers=(14,),
            rank=0,
        )


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_evidence_rejects_non_finite_scores(score: float) -> None:
    with pytest.raises(ValidationError):
        Evidence(
            evidence_id="evidence-1",
            question_id="question-1",
            document_id="report-2025",
            text="Revenue increased.",
            source_element_ids=("paragraph-10",),
            page_numbers=(14,),
            rank=1,
            retrieval_score=score,
        )


def test_evidence_serializes_tuples_as_json_arrays() -> None:
    evidence = Evidence(
        evidence_id="evidence-1",
        question_id="question-1",
        document_id="report-2025",
        text="Revenue increased.",
        source_element_ids=("paragraph-10",),
        page_numbers=(14,),
        rank=1,
    )

    result = evidence.model_dump(mode="json")

    assert result["source_element_ids"] == ["paragraph-10"]
    assert result["page_numbers"] == [14]
