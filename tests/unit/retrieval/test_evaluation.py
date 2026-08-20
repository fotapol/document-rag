"""Tests for source-lineage retrieval evaluation."""

import pytest

from document_rag.datasets.models import DatasetName
from document_rag.ingestion.chunking import DocumentChunk
from document_rag.retrieval.evaluation import (
    RetrievalEvaluationError,
    aggregate_evaluations,
    evaluate_query,
)
from document_rag.retrieval.models import QueryRetrievalEvaluation, RetrievalResult


def build_result(
    rank: int,
    source_element_ids: tuple[str, ...],
) -> RetrievalResult:
    """Build one synthetic ranked chunk."""

    text = f"retrieved chunk {rank}"
    return RetrievalResult(
        chunk=DocumentChunk(
            chunk_id=f"chunk-{rank}",
            document_id="document",
            document_sha256="a" * 64,
            filename="report.pdf",
            chunk_index=rank - 1,
            page_start=rank,
            page_end=rank,
            source_element_ids=source_element_ids,
            text=text,
            char_count=len(text),
            token_count=3,
            block_count=1,
        ),
        score=1.0 / rank,
        rank=rank,
    )


def evaluate(
    results: tuple[RetrievalResult, ...],
    *,
    gold_ids: tuple[str, ...] = ("gold-a",),
    example_id: str = "example",
) -> QueryRetrievalEvaluation:
    """Evaluate a synthetic FinQA result list."""

    return evaluate_query(
        dataset=DatasetName.FINQA,
        example_id=example_id,
        question="What was operating income?",
        gold_source_element_ids=gold_ids,
        results=results,
    )


def test_hit_and_recall_at_requested_cutoffs() -> None:
    """Hit and evidence recall must remain distinct for multiple gold IDs."""

    evaluation = evaluate_query(
        dataset=DatasetName.FINQA,
        example_id="multi-gold",
        question="Compare revenue and operating income.",
        gold_source_element_ids=("gold-a", "gold-b"),
        results=(
            build_result(1, ("unrelated",)),
            build_result(2, ("gold-a",)),
            build_result(3, ("another",)),
            build_result(4, ("gold-b",)),
        ),
    )

    assert evaluation.hit_at_k == ((1, 0), (3, 1), (5, 1))
    assert evaluation.recall_at_k == ((1, 0.0), (3, 0.5), (5, 1.0))
    assert evaluation.first_relevant_rank == 2


def test_mrr_is_one_when_first_result_is_relevant() -> None:
    """A rank-one hit contributes reciprocal rank one."""

    evaluation = evaluate((build_result(1, ("gold-a",)),))

    assert aggregate_evaluations((evaluation,)).mrr == 1.0


def test_mrr_uses_later_first_relevant_rank() -> None:
    """A first hit at rank three contributes one third."""

    evaluation = evaluate(
        (
            build_result(1, ("other-a",)),
            build_result(2, ("other-b",)),
            build_result(3, ("gold-a",)),
        )
    )

    assert aggregate_evaluations((evaluation,)).mrr == pytest.approx(1 / 3)


def test_mrr_is_zero_when_nothing_is_relevant() -> None:
    """A missed query should contribute zero reciprocal rank."""

    evaluation = evaluate((build_result(1, ("other",)),))

    assert aggregate_evaluations((evaluation,)).mrr == 0.0


def test_aggregate_metrics_macro_average_queries() -> None:
    """Dataset metrics should average query outcomes rather than evidence rows."""

    hit = evaluate(
        (build_result(1, ("gold-a",)),),
        example_id="hit",
    )
    miss = evaluate(
        (build_result(1, ("other",)),),
        example_id="miss",
    )
    metrics = aggregate_evaluations((hit, miss))

    assert metrics.hit_rate_at_k == ((1, 0.5), (3, 0.5), (5, 0.5))
    assert metrics.recall_at_k == ((1, 0.5), (3, 0.5), (5, 0.5))
    assert metrics.mrr == 0.5


def test_duplicate_evidence_across_chunks_does_not_inflate_recall() -> None:
    """Recall coverage should use a set of gold source IDs."""

    evaluation = evaluate_query(
        dataset=DatasetName.DOCFINQA,
        example_id="duplicate-evidence",
        question="What was revenue?",
        gold_source_element_ids=("gold-a", "gold-b"),
        results=(
            build_result(1, ("gold-a",)),
            build_result(2, ("gold-a",)),
        ),
        k_values=(2,),
    )

    assert evaluation.recall_at_k == ((2, 0.5),)


def test_query_without_gold_evidence_is_rejected() -> None:
    """Unevaluable questions must not dilute aggregate metrics."""

    with pytest.raises(RetrievalEvaluationError, match="evaluable gold"):
        evaluate_query(
            dataset=DatasetName.FINQA,
            example_id="missing-gold",
            question="What was revenue?",
            gold_source_element_ids=(),
            results=(),
        )


def test_duplicate_retrieved_chunks_are_rejected() -> None:
    """Repeated retrieval units would distort rank-based metrics."""

    result = build_result(1, ("gold-a",))
    duplicate = RetrievalResult(
        chunk=result.chunk,
        score=0.5,
        rank=2,
    )

    with pytest.raises(RetrievalEvaluationError, match="chunk IDs must be unique"):
        evaluate((result, duplicate))


def test_empty_evaluation_collection_is_rejected() -> None:
    """Aggregate metrics require at least one evaluable query."""

    with pytest.raises(RetrievalEvaluationError, match="At least one evaluable"):
        aggregate_evaluations(())
