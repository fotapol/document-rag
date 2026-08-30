"""Deterministic BM25 retrieval over normalized document chunks."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Literal

from rank_bm25 import BM25Okapi, BM25Plus  # type: ignore[import-untyped]

from document_rag.ingestion.chunking import DocumentChunk
from document_rag.retrieval.models import RetrievalResult

TOKENIZATION_STRATEGY = "unicode_casefold_financial_lexical_v1"
TABLE_QUERY_NORMALIZATION_STRATEGY = "financial_stopwords_deduplicated_v1"
_LEXICAL_TOKEN_PATTERN = re.compile(
    r"[^\W_]+(?:[.,][^\W_]+)*|[$€£¥%]",
    re.UNICODE,
)
_QUERY_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "much",
        "of",
        "on",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "which",
        "who",
        "with",
    }
)


def lexical_tokenize(text: str) -> tuple[str, ...]:
    """Case-fold words, decimals, and currency/percent signs into stable terms.

    The baseline intentionally omits stemming and semantic normalization. It
    keeps decimal-like values together and emits common financial symbols as
    separate terms while ignoring structural punctuation.
    """

    return tuple(_LEXICAL_TOKEN_PATTERN.findall(text.casefold()))


def normalize_bm25_query(query: str) -> tuple[str, ...]:
    """Remove query boilerplate and repeated terms without losing finance tokens.

    Values, currencies, percentages, years, and quarter labels are not stop
    words, so they pass through unchanged. If a question contains only stop
    words, the lexical tokens are retained rather than producing an empty query.
    """

    tokens = lexical_tokenize(query)
    informative_tokens = tuple(token for token in tokens if token not in _QUERY_STOP_WORDS)
    selected_tokens = informative_tokens or tokens
    return tuple(dict.fromkeys(selected_tokens))


class BM25Retriever:
    """In-memory BM25 index with deterministic tie-breaking.

    The established benchmark keeps Okapi and the original tokenizer defaults.
    Session-sized row indexes can opt into BM25+, whose positive inverse-
    document-frequency weights avoid penalizing common matches in tiny corpora.
    """

    def __init__(
        self,
        chunks: Iterable[DocumentChunk],
        *,
        variant: Literal["okapi", "plus"] = "okapi",
        query_tokenizer: Callable[[str], tuple[str, ...]] = lexical_tokenize,
    ) -> None:
        """Index unique chunks in stable identity order."""

        if variant not in {"okapi", "plus"}:
            raise ValueError("variant must be either 'okapi' or 'plus'.")

        materialized_chunks = tuple(chunks)
        chunk_ids = [chunk.chunk_id for chunk in materialized_chunks]

        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("BM25 chunks must have unique chunk IDs.")

        self._chunks = tuple(
            sorted(
                materialized_chunks,
                key=lambda chunk: (
                    chunk.document_id,
                    chunk.chunk_index,
                    chunk.chunk_id,
                ),
            )
        )

        tokenized_chunks = [lexical_tokenize(chunk.text) for chunk in self._chunks]

        if any(not tokens for tokens in tokenized_chunks):
            raise ValueError("Every BM25 chunk must contain at least one lexical token.")

        self._index = (
            (BM25Plus(tokenized_chunks) if variant == "plus" else BM25Okapi(tokenized_chunks))
            if tokenized_chunks
            else None
        )
        self._query_tokenizer = query_tokenizer

    @property
    def chunk_count(self) -> int:
        """Return the number of unique indexed chunks."""

        return len(self._chunks)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> tuple[RetrievalResult, ...]:
        """Return up to ``top_k`` chunks ordered by score and stable ID."""

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_tokens = self._query_tokenizer(query)

        if not query_tokens:
            raise ValueError("query must contain at least one lexical token.")

        if self._index is None:
            return ()

        scores = self._index.get_scores(query_tokens)
        ranked_indexes = sorted(
            range(len(self._chunks)),
            key=lambda index: (
                -float(scores[index]),
                self._chunks[index].chunk_id,
            ),
        )[:top_k]

        return tuple(
            RetrievalResult(
                chunk=self._chunks[index],
                score=float(scores[index]),
                rank=rank,
            )
            for rank, index in enumerate(ranked_indexes, start=1)
        )
