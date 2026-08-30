"""Offline tests for canonical table-row retrieval units."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from document_rag.ingestion.chunking import DocumentChunk
from document_rag.retrieval.bm25 import BM25Retriever, normalize_bm25_query
from document_rag.retrieval.dense import DenseRetriever, FloatMatrix
from document_rag.retrieval.hybrid import ReciprocalRankFusionRetriever
from document_rag.retrieval.models import RetrievalResult
from document_rag.retrieval.table_units import (
    ParentAwareRetriever,
    build_table_retrieval_corpus,
)

_TABLE = """# Quarterly Revenue Detail

Revenue is reported by quarter and geography.

<table>
<thead><tr><th>Quarter</th><th>Region</th><th>Revenue</th></tr></thead>
<tbody>
<tr><td>2024 Q1</td><td>Europe</td><td>$200,000</td></tr>
<tr><td>2024 Q4</td><td>North America</td><td>$210,000</td></tr>
<tr><td>Quarter</td><td>Region</td><td>Revenue</td></tr>
<tr><td>2024 Q1</td><td>North America</td><td>$220,000</td></tr>
<tr><td>2024 Q1</td><td>North America</td><td>$220,000</td></tr>
</tbody>
</table>"""


@dataclass(frozen=True)
class KeywordEmbedder:
    """Map diagnostic terms to deterministic offline vectors."""

    model_id: str = "fake/table-keywords"
    model_revision: str | None = "test"
    dimension: int = 5
    device: str = "cpu"

    def embed_documents(self, texts: Sequence[str], *, batch_size: int) -> FloatMatrix:
        assert batch_size > 0
        return self._embed(texts)

    def embed_queries(self, texts: Sequence[str], *, batch_size: int) -> FloatMatrix:
        assert batch_size > 0
        return self._embed(texts)

    def _embed(self, texts: Sequence[str]) -> FloatMatrix:
        return np.asarray(
            [
                [
                    1.0,
                    float("2024" in text.casefold()),
                    float("q1" in text.casefold()),
                    float("north america" in text.casefold()),
                    float("220,000" in text.casefold()),
                ]
                for text in texts
            ],
            dtype=np.float32,
        )


def test_html_rows_are_canonical_deduplicated_and_lineaged() -> None:
    """Rows should carry headers, title, page, row identity, and parent lineage."""

    parent = _chunk("chunk:parent", _TABLE, page=3)
    corpus = build_table_retrieval_corpus((parent,))
    rows = tuple(unit for unit in corpus.units if unit.retrieval_unit_kind == "table_row")
    narratives = tuple(unit for unit in corpus.units if unit.retrieval_unit_kind == "narrative")

    assert len(rows) == 3
    assert len({row.chunk_id for row in rows}) == 3
    assert len(narratives) == 1
    assert all(row.page_start == row.page_end == 3 for row in rows)
    assert all(row.source_element_ids[0].startswith("source:table-row:") for row in rows)
    assert all(row.parent_source_element_ids == parent.source_element_ids for row in rows)
    gold = next(row for row in rows if "$220,000" in row.text)
    assert gold.text == (
        "Table: Quarterly Revenue Detail\n"
        "Quarter: 2024 Q1 | Region: North America | Revenue: $220,000"
    )
    assert gold.parent_chunk_id == parent.chunk_id
    assert gold.parent_source_element_ids == parent.source_element_ids
    assert corpus.parent_for(gold) == parent
    assert "Revenue is reported by quarter and geography." in narratives[0].text
    assert "<table>" not in narratives[0].text


def test_narrative_from_a_table_chunk_remains_retrievable() -> None:
    """Row expansion must not remove adjacent prose needed by narrative questions."""

    corpus = build_table_retrieval_corpus((_chunk("chunk:parent", _TABLE),))
    retriever = BM25Retriever(
        corpus.units,
        variant="plus",
        query_tokenizer=normalize_bm25_query,
    )

    result = retriever.search("How is revenue reported by geography?", top_k=1)[0]

    assert result.chunk.retrieval_unit_kind == "narrative"
    assert "reported by quarter and geography" in result.chunk.text


def test_repeated_table_fragment_rows_are_deduplicated_deterministically() -> None:
    """Repeated headers and rows from split tables must not create duplicate results."""

    first_parent = _chunk("chunk:fragment-a", _TABLE, page=3)
    second_parent = _chunk("chunk:fragment-b", _TABLE, page=3)
    first = build_table_retrieval_corpus((first_parent, second_parent))
    second = build_table_retrieval_corpus((second_parent, first_parent))
    first_rows = tuple(unit for unit in first.units if unit.retrieval_unit_kind == "table_row")
    second_rows = tuple(unit for unit in second.units if unit.retrieval_unit_kind == "table_row")

    assert len(first_rows) == 3
    assert [row.chunk_id for row in first_rows] == [row.chunk_id for row in second_rows]
    assert [row.text for row in first_rows] == [row.text for row in second_rows]


def test_markdown_table_rows_include_title_and_columns() -> None:
    """Pipe tables should produce the same canonical row shape as HTML tables."""

    parent = _chunk(
        "chunk:markdown",
        "# Margins\n\n| Year | Gross margin |\n| --- | ---: |\n| 2024 | 42.5% |\n",
        page=7,
    )
    corpus = build_table_retrieval_corpus((parent,))

    assert len(corpus.units) == 1
    assert corpus.units[0].text == "Table: Margins\nYear: 2024 | Gross margin: 42.5%"
    assert corpus.units[0].retrieval_unit_kind == "table_row"


def test_bm25_dense_and_rrf_rank_the_diagnostic_row_first() -> None:
    """All retrievers should prefer the exact 2024 Q1 North America row."""

    parent = _chunk("chunk:parent", _TABLE, page=3)
    corpus = build_table_retrieval_corpus((parent,))
    query = "What was North America revenue in 2024 Q1?"
    bm25 = BM25Retriever(
        corpus.units,
        variant="plus",
        query_tokenizer=normalize_bm25_query,
    )
    dense = DenseRetriever(corpus.units, embedder=KeywordEmbedder(), batch_size=8)
    hybrid = ReciprocalRankFusionRetriever(
        lexical_retriever=bm25,
        semantic_retriever=dense,
        candidate_k=len(corpus.units),
    )

    bm25_results = bm25.search(query, top_k=len(corpus.units))
    dense_results = dense.search(query, top_k=len(corpus.units))
    hybrid_results = hybrid.search(query, top_k=len(corpus.units))

    assert "$220,000" in bm25_results[0].chunk.text
    assert "$220,000" in dense_results[0].chunk.text
    assert "$220,000" in hybrid_results[0].chunk.text
    assert len({result.chunk_id for result in hybrid_results}) == len(hybrid_results)
    assert hybrid.search(query, top_k=len(corpus.units)) == hybrid_results


def test_terms_from_separate_rows_never_form_one_retrieval_unit() -> None:
    """A row child must not combine a period from one row with another region."""

    table_without_gold = _TABLE.replace(
        "<tr><td>2024 Q1</td><td>North America</td><td>$220,000</td></tr>",
        "",
    )
    corpus = build_table_retrieval_corpus((_chunk("chunk:parent", table_without_gold),))
    rows = tuple(unit for unit in corpus.units if unit.retrieval_unit_kind == "table_row")

    assert rows
    assert not any("2024 Q1" in row.text and "North America" in row.text for row in rows)


def test_parent_aware_retriever_recovers_the_original_table_chunk() -> None:
    """A retrieved child should support explicit lookup of its full parent context."""

    parent = _chunk("chunk:parent", _TABLE)
    corpus = build_table_retrieval_corpus((parent,))
    row = next(unit for unit in corpus.units if "$220,000" in unit.text)
    result = RetrievalResult(chunk=row, score=1.0, rank=1)
    retriever = ParentAwareRetriever(
        retriever=_StaticRetriever((result,)),
        corpus=corpus,
    )

    retrieved = retriever.search("revenue", top_k=1)[0]

    assert retriever.parent_for(retrieved.chunk) == parent
    assert retrieved.to_record()["parent_chunk_id"] == parent.chunk_id


@dataclass(frozen=True)
class _StaticRetriever:
    results: tuple[RetrievalResult, ...]

    def search(self, query: str, *, top_k: int = 5) -> tuple[RetrievalResult, ...]:
        del query
        return self.results[:top_k]


def _chunk(chunk_id: str, text: str, *, page: int = 1) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="document:diagnostic",
        document_sha256="a" * 64,
        filename="financial_report.pdf",
        chunk_index=0,
        page_start=page,
        page_end=page,
        source_element_ids=("source:table",),
        text=text,
        char_count=len(text),
        token_count=len(text.split()),
        block_count=2,
    )
