"""Deterministic parent-aware diversification of ranked retrieval results."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from document_rag.ingestion.chunking import DocumentChunk
from document_rag.retrieval.models import RetrievalResult

DEFAULT_MAX_TABLE_ROWS_PER_PARENT = 2


class RankedRetriever(Protocol):
    """Minimal ranked-search interface accepted by the diversity wrapper."""

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> tuple[RetrievalResult, ...]:
        """Return a deterministic ranking for one query."""


class ParentDiverseRetriever:
    """Limit table-row crowding after retrieval without changing source scores."""

    def __init__(
        self,
        *,
        retriever: RankedRetriever,
        candidate_k: int,
        max_table_rows_per_parent: int = DEFAULT_MAX_TABLE_ROWS_PER_PARENT,
    ) -> None:
        """Store the underlying ranking and deterministic selection limits."""

        _validate_positive(candidate_k, label="candidate_k")
        _validate_positive(
            max_table_rows_per_parent,
            label="max_table_rows_per_parent",
        )
        self._retriever = retriever
        self._candidate_k = candidate_k
        self._max_table_rows_per_parent = max_table_rows_per_parent

    @property
    def candidate_k(self) -> int:
        """Return the ranking depth inspected before diversification."""

        return self._candidate_k

    @property
    def max_table_rows_per_parent(self) -> int:
        """Return the maximum selected rows from one logical parent source."""

        return self._max_table_rows_per_parent

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> tuple[RetrievalResult, ...]:
        """Greedily retain ranked candidates while capping sibling table rows."""

        if not query.strip():
            raise ValueError("query must not be empty.")

        _validate_positive(top_k, label="top_k")
        candidates = self._retriever.search(
            query,
            top_k=max(top_k, self._candidate_k),
        )
        _validate_ranking(candidates)
        selected: list[RetrievalResult] = []
        row_counts: dict[tuple[str, tuple[str, ...]], int] = {}

        for candidate in candidates:
            parent_key = table_parent_key(candidate.chunk)

            if parent_key is not None:
                row_count = row_counts.get(parent_key, 0)

                if row_count >= self._max_table_rows_per_parent:
                    continue

                row_counts[parent_key] = row_count + 1

            selected.append(replace(candidate, rank=len(selected) + 1))

            if len(selected) == top_k:
                break

        return tuple(selected)


def table_parent_key(chunk: DocumentChunk) -> tuple[str, tuple[str, ...]] | None:
    """Return a stable logical-parent key for a canonical table-row unit."""

    if chunk.retrieval_unit_kind != "table_row":
        return None

    lineage = chunk.parent_source_element_ids

    if not lineage:
        lineage = (chunk.parent_chunk_id or chunk.chunk_id,)

    return chunk.document_id, lineage


def _validate_ranking(results: tuple[RetrievalResult, ...]) -> None:
    expected_ranks = tuple(range(1, len(results) + 1))

    if tuple(result.rank for result in results) != expected_ranks:
        raise ValueError("candidate ranks must be unique and contiguous from 1.")

    chunk_ids = tuple(result.chunk_id for result in results)

    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("candidate chunk IDs must be unique.")


def _validate_positive(value: int, *, label: str) -> None:
    if value <= 0:
        raise ValueError(f"{label} must be positive.")
