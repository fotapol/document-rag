"""Offline tests for deterministic parent-aware result diversification."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from document_rag.ingestion.chunking import DocumentChunk, RetrievalUnitKind
from document_rag.retrieval.diversity import ParentDiverseRetriever, table_parent_key
from document_rag.retrieval.hybrid import HybridRetrievalResult
from document_rag.retrieval.models import RetrievalResult


@dataclass
class _StaticRetriever:
    results: tuple[RetrievalResult, ...]
    requests: list[tuple[str, int]] = field(default_factory=list)

    def search(self, query: str, *, top_k: int = 5) -> tuple[RetrievalResult, ...]:
        self.requests.append((query, top_k))
        return self.results[:top_k]


def test_sibling_rows_are_capped_and_deeper_sources_are_promoted() -> None:
    """One table must not occupy the full grounded-context result list."""

    candidates = (
        _result(_row("row:a1", "fragment:a", ("source:table-a",)), rank=1),
        _result(_row("row:a2", "fragment:a", ("source:table-a",)), rank=2),
        _result(_row("row:a3", "fragment:b", ("source:table-a",)), rank=3),
        _result(_chunk("narrative:annual"), rank=4),
        _result(_row("row:b1", "fragment:c", ("source:table-b",)), rank=5),
        _result(_chunk("chunk:comparison"), rank=6),
    )
    underlying = _StaticRetriever(candidates)
    retriever = ParentDiverseRetriever(
        retriever=underlying,
        candidate_k=6,
        max_table_rows_per_parent=2,
    )

    results = retriever.search("compare annual revenue", top_k=5)

    assert underlying.requests == [("compare annual revenue", 6)]
    assert [result.chunk_id for result in results] == [
        "row:a1",
        "row:a2",
        "narrative:annual",
        "row:b1",
        "chunk:comparison",
    ]
    assert [result.rank for result in results] == [1, 2, 3, 4, 5]
    assert [result.score for result in results] == [0.99, 0.98, 0.96, 0.95, 0.94]
    assert all(isinstance(result, HybridRetrievalResult) for result in results)
    promoted = results[2]
    assert isinstance(promoted, HybridRetrievalResult)
    assert promoted.bm25_rank == 4
    assert promoted.dense_rank == 4


def test_split_fragments_share_a_logical_parent_lineage() -> None:
    """Fragment chunk IDs must not bypass the configured logical-source cap."""

    first = _row("row:one", "fragment:one", ("heading:1", "table:1"))
    second = _row("row:two", "fragment:two", ("heading:1", "table:1"))

    assert first.parent_chunk_id != second.parent_chunk_id
    assert table_parent_key(first) == table_parent_key(second)


def test_narrative_units_are_not_counted_as_table_rows() -> None:
    """Adjacent prose should remain eligible even when its table rows reach the cap."""

    lineage = ("source:shared",)
    candidates = (
        _result(_row("row:1", "parent:1", lineage), rank=1),
        _result(_row("row:2", "parent:1", lineage), rank=2),
        _result(_narrative("narrative:1", "parent:1", lineage), rank=3),
    )
    retriever = ParentDiverseRetriever(
        retriever=_StaticRetriever(candidates),
        candidate_k=3,
        max_table_rows_per_parent=1,
    )

    results = retriever.search("revenue", top_k=2)

    assert [result.chunk_id for result in results] == ["row:1", "narrative:1"]
    assert table_parent_key(candidates[2].chunk) is None


def test_selection_is_deterministic_and_duplicate_free() -> None:
    """Identical candidates and configuration must produce identical rankings."""

    candidates = tuple(
        _result(_row(f"row:{index}", "parent:1", ("source:table",)), rank=index)
        for index in range(1, 5)
    )
    retriever = ParentDiverseRetriever(
        retriever=_StaticRetriever(candidates),
        candidate_k=4,
        max_table_rows_per_parent=2,
    )

    first = retriever.search("revenue", top_k=4)
    second = retriever.search("revenue", top_k=4)

    assert first == second
    assert [result.chunk_id for result in first] == ["row:1", "row:2"]
    assert len({result.chunk_id for result in first}) == len(first)


@pytest.mark.parametrize("value", [0, -1])
def test_positive_configuration_is_required(value: int) -> None:
    """Invalid selection limits should fail before any retrieval work."""

    with pytest.raises(ValueError, match="candidate_k"):
        ParentDiverseRetriever(retriever=_StaticRetriever(()), candidate_k=value)

    with pytest.raises(ValueError, match="max_table_rows_per_parent"):
        ParentDiverseRetriever(
            retriever=_StaticRetriever(()),
            candidate_k=1,
            max_table_rows_per_parent=value,
        )


def _result(chunk: DocumentChunk, *, rank: int) -> HybridRetrievalResult:
    return HybridRetrievalResult(
        chunk=chunk,
        score=1.0 - (rank / 100),
        rank=rank,
        bm25_rank=rank,
        dense_rank=rank,
    )


def _row(
    chunk_id: str,
    parent_chunk_id: str,
    parent_source_element_ids: tuple[str, ...],
) -> DocumentChunk:
    return _chunk(
        chunk_id,
        retrieval_unit_kind="table_row",
        parent_chunk_id=parent_chunk_id,
        parent_source_element_ids=parent_source_element_ids,
    )


def _narrative(
    chunk_id: str,
    parent_chunk_id: str,
    parent_source_element_ids: tuple[str, ...],
) -> DocumentChunk:
    return _chunk(
        chunk_id,
        retrieval_unit_kind="narrative",
        parent_chunk_id=parent_chunk_id,
        parent_source_element_ids=parent_source_element_ids,
    )


def _chunk(
    chunk_id: str,
    *,
    retrieval_unit_kind: RetrievalUnitKind = "chunk",
    parent_chunk_id: str | None = None,
    parent_source_element_ids: tuple[str, ...] = (),
) -> DocumentChunk:
    text = f"Evidence in {chunk_id}."
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="document:test",
        document_sha256="a" * 64,
        filename="report.pdf",
        chunk_index=0,
        page_start=1,
        page_end=1,
        source_element_ids=(f"source:{chunk_id}",),
        text=text,
        char_count=len(text),
        token_count=len(text.split()),
        block_count=1,
        retrieval_unit_kind=retrieval_unit_kind,
        parent_chunk_id=parent_chunk_id,
        parent_source_element_ids=parent_source_element_ids,
    )
