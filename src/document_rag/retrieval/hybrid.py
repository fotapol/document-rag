"""Deterministic reciprocal-rank fusion for lexical and dense retrieval."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from document_rag.ingestion.chunking import DocumentChunk
from document_rag.retrieval.models import RetrievalResult

DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_K = 20
RRF_FORMULA = "sum(1 / (rrf_k + component_rank))"


class RankedRetriever(Protocol):
    """Minimal ranked-search interface shared by BM25 and dense retrieval."""

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> tuple[RetrievalResult, ...]:
        """Return a ranked result list for one query."""


@dataclass(frozen=True, slots=True)
class HybridRetrievalResult(RetrievalResult):
    """One fused result with its optional lexical and semantic ranks."""

    bm25_rank: int | None
    dense_rank: int | None

    def __post_init__(self) -> None:
        """Validate the final rank, RRF score, and component ranks."""

        RetrievalResult.__post_init__(self)

        if self.bm25_rank is None and self.dense_rank is None:
            raise ValueError("A hybrid result must have at least one component rank.")

        if self.bm25_rank is not None and self.bm25_rank <= 0:
            raise ValueError("bm25_rank must be positive when present.")

        if self.dense_rank is not None and self.dense_rank <= 0:
            raise ValueError("dense_rank must be positive when present.")

    def to_record(self) -> dict[str, object]:
        """Return generic result fields plus compact fusion diagnostics."""

        record = RetrievalResult.to_record(self)
        record["rrf_score"] = self.score

        if self.bm25_rank is not None:
            record["bm25_rank"] = self.bm25_rank

        if self.dense_rank is not None:
            record["dense_rank"] = self.dense_rank

        return record


class ReciprocalRankFusionRetriever:
    """Fuse BM25 and dense candidate rankings without combining raw scores."""

    def __init__(
        self,
        *,
        lexical_retriever: RankedRetriever,
        semantic_retriever: RankedRetriever,
        rrf_k: int = DEFAULT_RRF_K,
        candidate_k: int = DEFAULT_CANDIDATE_K,
    ) -> None:
        """Store two frozen retrievers and validate fusion configuration."""

        _validate_rrf_k(rrf_k)
        _validate_positive(candidate_k, label="candidate_k")
        self._lexical_retriever = lexical_retriever
        self._semantic_retriever = semantic_retriever
        self._rrf_k = rrf_k
        self._candidate_k = candidate_k

    @property
    def rrf_k(self) -> int:
        """Return the reciprocal-rank denominator constant."""

        return self._rrf_k

    @property
    def candidate_k(self) -> int:
        """Return the per-component candidate depth."""

        return self._candidate_k

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> tuple[HybridRetrievalResult, ...]:
        """Retrieve component candidates and return their deterministic fusion."""

        if not query.strip():
            raise ValueError("query must not be empty.")

        _validate_positive(top_k, label="top_k")
        bm25_results = self._lexical_retriever.search(
            query,
            top_k=self._candidate_k,
        )
        dense_results = self._semantic_retriever.search(
            query,
            top_k=self._candidate_k,
        )
        return fuse_rankings(
            bm25_results=bm25_results,
            dense_results=dense_results,
            rrf_k=self._rrf_k,
            top_k=top_k,
        )


def fuse_rankings(
    *,
    bm25_results: Iterable[RetrievalResult],
    dense_results: Iterable[RetrievalResult],
    rrf_k: int = DEFAULT_RRF_K,
    top_k: int = 5,
) -> tuple[HybridRetrievalResult, ...]:
    """Deduplicate two rankings and score their union with unweighted RRF."""

    _validate_rrf_k(rrf_k)
    _validate_positive(top_k, label="top_k")
    ordered_bm25 = _validate_component_ranking(
        bm25_results,
        label="BM25",
    )
    ordered_dense = _validate_component_ranking(
        dense_results,
        label="dense",
    )
    chunks_by_id: dict[str, DocumentChunk] = {}
    bm25_ranks = _index_component_ranks(
        ordered_bm25,
        chunks_by_id=chunks_by_id,
        label="BM25",
    )
    dense_ranks = _index_component_ranks(
        ordered_dense,
        chunks_by_id=chunks_by_id,
        label="dense",
    )
    scored_candidates = tuple(
        (
            chunk_id,
            _rrf_score(
                bm25_rank=bm25_ranks.get(chunk_id),
                dense_rank=dense_ranks.get(chunk_id),
                rrf_k=rrf_k,
            ),
        )
        for chunk_id in sorted(chunks_by_id)
    )
    ranked_candidates = sorted(
        scored_candidates,
        key=lambda candidate: (-candidate[1], candidate[0]),
    )[:top_k]

    return tuple(
        HybridRetrievalResult(
            chunk=chunks_by_id[chunk_id],
            score=score,
            rank=rank,
            bm25_rank=bm25_ranks.get(chunk_id),
            dense_rank=dense_ranks.get(chunk_id),
        )
        for rank, (chunk_id, score) in enumerate(ranked_candidates, start=1)
    )


def _validate_component_ranking(
    results: Iterable[RetrievalResult],
    *,
    label: str,
) -> tuple[RetrievalResult, ...]:
    ordered = tuple(sorted(results, key=lambda result: result.rank))
    expected_ranks = tuple(range(1, len(ordered) + 1))

    if tuple(result.rank for result in ordered) != expected_ranks:
        raise ValueError(f"{label} candidate ranks must be unique and contiguous from 1.")

    chunk_ids = tuple(result.chunk_id for result in ordered)

    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError(f"{label} candidate chunk IDs must be unique.")

    return ordered


def _index_component_ranks(
    results: tuple[RetrievalResult, ...],
    *,
    chunks_by_id: dict[str, DocumentChunk],
    label: str,
) -> dict[str, int]:
    ranks: dict[str, int] = {}

    for result in results:
        existing_chunk = chunks_by_id.get(result.chunk_id)

        if existing_chunk is not None and existing_chunk != result.chunk:
            raise ValueError(
                f"{label} candidate {result.chunk_id!r} conflicts with the other component."
            )

        chunks_by_id[result.chunk_id] = result.chunk
        ranks[result.chunk_id] = result.rank

    return ranks


def _rrf_score(
    *,
    bm25_rank: int | None,
    dense_rank: int | None,
    rrf_k: int,
) -> float:
    return sum(1.0 / (rrf_k + rank) for rank in (bm25_rank, dense_rank) if rank is not None)


def _validate_rrf_k(rrf_k: int) -> None:
    _validate_positive(rrf_k, label="rrf_k")


def _validate_positive(value: int, *, label: str) -> None:
    if value <= 0:
        raise ValueError(f"{label} must be positive.")
