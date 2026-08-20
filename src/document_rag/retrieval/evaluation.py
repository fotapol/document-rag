"""Source-lineage-based retrieval metrics independent of any retriever."""

from __future__ import annotations

from collections.abc import Iterable

from document_rag.datasets.models import DatasetName
from document_rag.retrieval.models import (
    QueryRetrievalEvaluation,
    RetrievalMetrics,
    RetrievalResult,
)

DEFAULT_K_VALUES = (1, 3, 5)


class RetrievalEvaluationError(ValueError):
    """Raised when retrieval output cannot be evaluated without ambiguity."""


def evaluate_query(
    *,
    dataset: DatasetName,
    example_id: str,
    question: str,
    gold_source_element_ids: Iterable[str],
    results: Iterable[RetrievalResult],
    k_values: Iterable[int] = DEFAULT_K_VALUES,
) -> QueryRetrievalEvaluation:
    """Evaluate one ranked result list through exact source-ID intersections."""

    gold_ids = _normalize_gold_ids(gold_source_element_ids)
    selected_k_values = normalize_k_values(k_values)
    ordered_results = _validate_results(tuple(results))

    first_relevant_rank = next(
        (
            result.rank
            for result in ordered_results
            if gold_ids.intersection(result.chunk.source_element_ids)
        ),
        None,
    )
    hit_at_k: list[tuple[int, int]] = []
    recall_at_k: list[tuple[int, float]] = []

    for k in selected_k_values:
        covered_ids = {
            source_element_id
            for result in ordered_results[:k]
            for source_element_id in result.chunk.source_element_ids
        }
        relevant_ids = gold_ids & covered_ids

        hit_at_k.append((k, int(bool(relevant_ids))))
        recall_at_k.append((k, len(relevant_ids) / len(gold_ids)))

    return QueryRetrievalEvaluation(
        dataset=dataset,
        example_id=example_id,
        question=question,
        gold_source_element_ids=tuple(sorted(gold_ids)),
        results=ordered_results,
        first_relevant_rank=first_relevant_rank,
        hit_at_k=tuple(hit_at_k),
        recall_at_k=tuple(recall_at_k),
    )


def aggregate_evaluations(
    evaluations: Iterable[QueryRetrievalEvaluation],
) -> RetrievalMetrics:
    """Macro-average Hit Rate, evidence Recall, and reciprocal rank."""

    ordered_evaluations = tuple(
        sorted(
            evaluations,
            key=lambda evaluation: (
                evaluation.dataset.value,
                evaluation.example_id,
            ),
        )
    )

    if not ordered_evaluations:
        raise RetrievalEvaluationError("At least one evaluable query is required.")

    k_values = tuple(k for k, _ in ordered_evaluations[0].hit_at_k)

    for evaluation in ordered_evaluations:
        if tuple(k for k, _ in evaluation.hit_at_k) != k_values:
            raise RetrievalEvaluationError("All query evaluations must use the same k values.")

        if tuple(k for k, _ in evaluation.recall_at_k) != k_values:
            raise RetrievalEvaluationError("Hit and recall metrics must use the same k values.")

    query_count = len(ordered_evaluations)

    return RetrievalMetrics(
        query_count=query_count,
        hit_rate_at_k=tuple(
            (
                k,
                sum(dict(evaluation.hit_at_k)[k] for evaluation in ordered_evaluations)
                / query_count,
            )
            for k in k_values
        ),
        recall_at_k=tuple(
            (
                k,
                sum(dict(evaluation.recall_at_k)[k] for evaluation in ordered_evaluations)
                / query_count,
            )
            for k in k_values
        ),
        mrr=sum(
            0.0 if evaluation.first_relevant_rank is None else 1.0 / evaluation.first_relevant_rank
            for evaluation in ordered_evaluations
        )
        / query_count,
    )


def _normalize_gold_ids(values: Iterable[str]) -> frozenset[str]:
    gold_ids = frozenset(value.strip() for value in values if value.strip())

    if not gold_ids:
        raise RetrievalEvaluationError(
            "A query must contain at least one evaluable gold source element ID."
        )

    return gold_ids


def normalize_k_values(values: Iterable[int]) -> tuple[int, ...]:
    """Validate, sort, and freeze metric cutoffs."""

    materialized_values = tuple(values)

    if not materialized_values:
        raise RetrievalEvaluationError("At least one k value is required.")

    if any(value <= 0 for value in materialized_values):
        raise RetrievalEvaluationError("k values must be positive.")

    if len(set(materialized_values)) != len(materialized_values):
        raise RetrievalEvaluationError("k values must be unique.")

    return tuple(sorted(materialized_values))


def _validate_results(
    results: tuple[RetrievalResult, ...],
) -> tuple[RetrievalResult, ...]:
    ordered_results = tuple(sorted(results, key=lambda result: result.rank))
    expected_ranks = tuple(range(1, len(ordered_results) + 1))

    if tuple(result.rank for result in ordered_results) != expected_ranks:
        raise RetrievalEvaluationError("Result ranks must be unique and contiguous from 1.")

    chunk_ids = [result.chunk_id for result in ordered_results]

    if len(set(chunk_ids)) != len(chunk_ids):
        raise RetrievalEvaluationError("Retrieved chunk IDs must be unique.")

    return ordered_results
