"""Tests for deterministic BM25 chunk retrieval."""

import pytest

from document_rag.ingestion.chunking import DocumentChunk
from document_rag.retrieval.bm25 import (
    BM25Retriever,
    lexical_tokenize,
    normalize_bm25_query,
)


def build_chunk(
    chunk_id: str,
    text: str,
    *,
    source_element_ids: tuple[str, ...] = ("source",),
) -> DocumentChunk:
    """Build a minimal immutable retrieval chunk."""

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="document",
        document_sha256="a" * 64,
        filename="report.pdf",
        chunk_index=int(chunk_id.removeprefix("chunk-")),
        page_start=1,
        page_end=1,
        source_element_ids=source_element_ids,
        text=text,
        char_count=len(text),
        token_count=len(lexical_tokenize(text)),
        block_count=1,
    )


def build_chunks() -> tuple[DocumentChunk, ...]:
    """Build a corpus with one exact operating-income match."""

    return (
        build_chunk("chunk-1", "Cash and cash equivalents declined."),
        build_chunk(
            "chunk-2",
            "Operating income was $2.6M in 2025.",
            source_element_ids=("operating-income",),
        ),
        build_chunk("chunk-3", "Share-based compensation expense increased."),
    )


def test_lexical_tokenizer_casefolds_financial_terms() -> None:
    """The lexical baseline should preserve decimals and financial symbols."""

    assert lexical_tokenize("Operating Income: $2.6M") == (
        "operating",
        "income",
        "$",
        "2.6m",
    )


def test_query_normalization_removes_boilerplate_and_duplicate_terms() -> None:
    """BM25 should count each meaningful question term only once."""

    assert normalize_bm25_query("What was revenue revenue in 2024 Q1 in North America?") == (
        "revenue",
        "2024",
        "q1",
        "north",
        "america",
    )


def test_query_normalization_preserves_financial_values_and_units() -> None:
    """Currency, percentages, decimals, and units must remain searchable."""

    assert normalize_bm25_query("What was $2.6M as a percentage, 12.5%?") == (
        "$",
        "2.6m",
        "percentage",
        "12.5",
        "%",
    )


def test_query_normalization_keeps_stop_words_as_an_all_stopword_fallback() -> None:
    """A grammatical-only query should remain valid instead of becoming empty."""

    assert normalize_bm25_query("What was it?") == ("what", "was", "it")


def test_exact_lexical_match_ranks_relevant_chunk_first() -> None:
    """An exact financial phrase should rank above unrelated chunks."""

    results = BM25Retriever(build_chunks()).search(
        "What was operating income in 2025?",
        top_k=3,
    )

    assert results[0].chunk_id == "chunk-2"
    assert results[0].chunk.source_element_ids == ("operating-income",)
    assert results[0].score > results[1].score


def test_results_are_deterministic_for_reordered_input() -> None:
    """Input iteration order must not affect scores or tie ordering."""

    chunks = build_chunks()
    first = BM25Retriever(chunks).search("income", top_k=3)
    second = BM25Retriever(reversed(chunks)).search("income", top_k=3)

    assert first == second


def test_ranking_contains_no_duplicate_chunks() -> None:
    """Every indexed chunk should occur at most once in a ranking."""

    results = BM25Retriever(build_chunks()).search("income", top_k=3)

    assert len({result.chunk_id for result in results}) == len(results)


def test_top_k_is_respected() -> None:
    """The requested result cutoff should bound the ranking."""

    results = BM25Retriever(build_chunks()).search("income", top_k=2)

    assert len(results) == 2
    assert [result.rank for result in results] == [1, 2]


def test_top_k_larger_than_corpus_returns_every_chunk() -> None:
    """A large cutoff should not duplicate or pad results."""

    results = BM25Retriever(build_chunks()).search("income", top_k=10)

    assert len(results) == 3
    assert [result.rank for result in results] == [1, 2, 3]


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_is_rejected(top_k: int) -> None:
    """Non-positive cutoffs are configuration errors."""

    with pytest.raises(ValueError, match="top_k must be positive"):
        BM25Retriever(build_chunks()).search("income", top_k=top_k)


def test_empty_index_returns_no_results() -> None:
    """A valid query against an empty index should return an empty ranking."""

    assert BM25Retriever(()).search("income", top_k=5) == ()


def test_empty_query_is_rejected() -> None:
    """Queries without lexical terms must not produce arbitrary tie results."""

    with pytest.raises(ValueError, match="query must contain"):
        BM25Retriever(build_chunks()).search(" : -- ", top_k=3)


def test_duplicate_chunk_ids_are_rejected() -> None:
    """Duplicate identities would make ranking output ambiguous."""

    chunk = build_chunk("chunk-1", "Revenue increased.")

    with pytest.raises(ValueError, match="unique chunk IDs"):
        BM25Retriever((chunk, chunk))
