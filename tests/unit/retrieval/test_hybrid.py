"""Offline tests for deterministic reciprocal-rank fusion."""

from dataclasses import dataclass, field

import pytest

from document_rag.datasets.models import DatasetName
from document_rag.ingestion.chunking import DocumentChunk
from document_rag.retrieval.benchmark import serialize_evaluations_jsonl
from document_rag.retrieval.evaluation import evaluate_query
from document_rag.retrieval.hybrid import (
    HybridRetrievalResult,
    ReciprocalRankFusionRetriever,
    fuse_rankings,
)
from document_rag.retrieval.models import RetrievalResult


@dataclass(slots=True)
class FakeRetriever:
    """Return controlled candidates while recording requested depths."""

    results: tuple[RetrievalResult, ...]
    requested_top_k: list[int] = field(default_factory=list)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> tuple[RetrievalResult, ...]:
        assert query
        self.requested_top_k.append(top_k)
        return self.results[:top_k]


def build_chunk(chunk_id: str, *, source_id: str | None = None) -> DocumentChunk:
    """Build one immutable synthetic chunk with source lineage."""

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="document",
        document_sha256="a" * 64,
        filename="report.pdf",
        chunk_index=ord(chunk_id[-1]) if chunk_id else 0,
        page_start=1,
        page_end=1,
        source_element_ids=(source_id or f"source-{chunk_id}",),
        text=f"Text for {chunk_id}",
        char_count=len(f"Text for {chunk_id}"),
        token_count=3,
        block_count=1,
    )


def rank_chunks(*chunk_ids: str) -> tuple[RetrievalResult, ...]:
    """Build a contiguous controlled component ranking."""

    return tuple(
        RetrievalResult(
            chunk=build_chunk(chunk_id),
            score=float(len(chunk_ids) - rank),
            rank=rank,
        )
        for rank, chunk_id in enumerate(chunk_ids, start=1)
    )


def test_shared_chunk_receives_two_rrf_contributions_and_is_deduplicated() -> None:
    """A candidate present in both rankings should receive both rank terms once."""

    results = fuse_rankings(
        bm25_results=rank_chunks("chunk-a", "chunk-b"),
        dense_results=rank_chunks("chunk-b", "chunk-c"),
        rrf_k=60,
        top_k=5,
    )

    shared = next(result for result in results if result.chunk_id == "chunk-b")
    assert len(results) == 3
    assert shared.score == pytest.approx(1 / 62 + 1 / 61)
    assert shared.bm25_rank == 2
    assert shared.dense_rank == 1


def test_single_component_chunk_receives_one_contribution() -> None:
    """A component-exclusive candidate should receive exactly one rank term."""

    results = fuse_rankings(
        bm25_results=rank_chunks("chunk-a"),
        dense_results=(),
        rrf_k=60,
    )

    assert results[0].score == pytest.approx(1 / 61)
    assert results[0].bm25_rank == 1
    assert results[0].dense_rank is None


def test_final_order_follows_rrf_score() -> None:
    """The candidate with contributions from both components should rank first."""

    results = fuse_rankings(
        bm25_results=rank_chunks("chunk-a", "chunk-b", "chunk-c"),
        dense_results=rank_chunks("chunk-c", "chunk-b", "chunk-d"),
        rrf_k=60,
        top_k=4,
    )

    assert [result.chunk_id for result in results[:2]] == ["chunk-c", "chunk-b"]
    assert [result.rank for result in results] == [1, 2, 3, 4]
    assert all(
        results[index].score >= results[index + 1].score for index in range(len(results) - 1)
    )


def test_equal_rrf_scores_use_chunk_id_tie_breaking() -> None:
    """Equal scores should use ascending stable chunk identity."""

    results = fuse_rankings(
        bm25_results=rank_chunks("chunk-z"),
        dense_results=rank_chunks("chunk-a"),
    )

    assert [result.chunk_id for result in results] == ["chunk-a", "chunk-z"]


def test_top_k_is_respected() -> None:
    """Fusion should truncate its unique candidate union to the requested depth."""

    results = fuse_rankings(
        bm25_results=rank_chunks("chunk-a", "chunk-b", "chunk-c"),
        dense_results=rank_chunks("chunk-d", "chunk-e", "chunk-f"),
        top_k=2,
    )

    assert len(results) == 2


def test_retriever_requests_configured_candidate_depth() -> None:
    """Component searches should use candidate_k rather than final top_k."""

    lexical = FakeRetriever(rank_chunks("chunk-a", "chunk-b", "chunk-c"))
    semantic = FakeRetriever(rank_chunks("chunk-d", "chunk-e", "chunk-f"))
    retriever = ReciprocalRankFusionRetriever(
        lexical_retriever=lexical,
        semantic_retriever=semantic,
        candidate_k=2,
    )

    results = retriever.search("financial question", top_k=1)

    assert len(results) == 1
    assert lexical.requested_top_k == [2]
    assert semantic.requested_top_k == [2]


@pytest.mark.parametrize("rrf_k", [0, -1])
def test_invalid_rrf_k_is_rejected(rrf_k: int) -> None:
    """The reciprocal-rank denominator constant must be positive."""

    with pytest.raises(ValueError, match="rrf_k must be positive"):
        ReciprocalRankFusionRetriever(
            lexical_retriever=FakeRetriever(()),
            semantic_retriever=FakeRetriever(()),
            rrf_k=rrf_k,
        )


@pytest.mark.parametrize("candidate_k", [0, -1])
def test_invalid_candidate_depth_is_rejected(candidate_k: int) -> None:
    """Per-component candidate depth must be positive."""

    with pytest.raises(ValueError, match="candidate_k must be positive"):
        ReciprocalRankFusionRetriever(
            lexical_retriever=FakeRetriever(()),
            semantic_retriever=FakeRetriever(()),
            candidate_k=candidate_k,
        )


def test_source_lineage_and_existing_result_model_are_preserved() -> None:
    """Fused results should retain the complete original chunk reference."""

    chunk = build_chunk("chunk-a", source_id="gold-source")
    component_result = RetrievalResult(chunk=chunk, score=7.0, rank=1)
    results = fuse_rankings(
        bm25_results=(component_result,),
        dense_results=(),
    )

    assert isinstance(results[0], RetrievalResult)
    assert isinstance(results[0], HybridRetrievalResult)
    assert results[0].chunk is chunk
    assert results[0].chunk.source_element_ids == ("gold-source",)


def test_identical_rankings_produce_identical_output() -> None:
    """Repeated fusion of the same component lists should be deterministic."""

    bm25 = rank_chunks("chunk-a", "chunk-b")
    dense = rank_chunks("chunk-b", "chunk-c")

    first = fuse_rankings(bm25_results=bm25, dense_results=dense)
    second = fuse_rankings(bm25_results=bm25, dense_results=dense)

    assert first == second


@pytest.mark.parametrize("empty_component", ["bm25", "dense"])
def test_one_empty_component_ranking_is_handled(empty_component: str) -> None:
    """Fusion should degrade to reciprocal ranks from the non-empty component."""

    populated = rank_chunks("chunk-a", "chunk-b")
    results = fuse_rankings(
        bm25_results=() if empty_component == "bm25" else populated,
        dense_results=populated if empty_component == "bm25" else (),
    )

    assert [result.chunk_id for result in results] == ["chunk-a", "chunk-b"]


def test_both_empty_component_rankings_return_empty_output() -> None:
    """No component candidates should explicitly produce no fused candidates."""

    assert fuse_rankings(bm25_results=(), dense_results=()) == ()


def test_existing_evaluator_accepts_hybrid_results_unchanged() -> None:
    """Source-lineage metrics should consume fused results without new logic."""

    chunk = build_chunk("chunk-a", source_id="gold-source")
    results = fuse_rankings(
        bm25_results=(RetrievalResult(chunk=chunk, score=1.0, rank=1),),
        dense_results=(),
    )
    evaluation = evaluate_query(
        dataset=DatasetName.FINQA,
        example_id="hybrid-example",
        question="financial question",
        gold_source_element_ids=("gold-source",),
        results=results,
    )

    assert evaluation.hit_at_k == ((1, 1), (3, 1), (5, 1))
    assert evaluation.recall_at_k == ((1, 1.0), (3, 1.0), (5, 1.0))
    assert evaluation.first_relevant_rank == 1


def test_hybrid_prediction_serialization_is_deterministic_and_diagnostic() -> None:
    """Stable JSONL should expose component ranks and the final RRF score."""

    results = fuse_rankings(
        bm25_results=rank_chunks("chunk-a", "chunk-b"),
        dense_results=rank_chunks("chunk-b", "chunk-c"),
    )
    evaluation = evaluate_query(
        dataset=DatasetName.FINQA,
        example_id="hybrid-example",
        question="financial question",
        gold_source_element_ids=("source-chunk-b",),
        results=results,
    )

    first = serialize_evaluations_jsonl((evaluation,))
    second = serialize_evaluations_jsonl((evaluation,))

    assert first == second
    assert b'"bm25_rank"' in first
    assert b'"dense_rank"' in first
    assert b'"rrf_score"' in first
