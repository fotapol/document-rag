"""Deterministic BM25 retrieval over normalized document chunks."""

from __future__ import annotations

import re
from collections.abc import Iterable

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from document_rag.ingestion.chunking import DocumentChunk
from document_rag.retrieval.models import RetrievalResult

TOKENIZATION_STRATEGY = "unicode_casefold_financial_lexical_v1"
_LEXICAL_TOKEN_PATTERN = re.compile(
    r"[^\W_]+(?:[.,][^\W_]+)*|[$€£¥%]",
    re.UNICODE,
)


def lexical_tokenize(text: str) -> tuple[str, ...]:
    """Case-fold words, decimals, and currency/percent signs into stable terms.

    The baseline intentionally omits stemming and semantic normalization. It
    keeps decimal-like values together and emits common financial symbols as
    separate terms while ignoring structural punctuation.
    """

    return tuple(_LEXICAL_TOKEN_PATTERN.findall(text.casefold()))


class BM25Retriever:
    """In-memory BM25Okapi index with deterministic tie-breaking."""

    def __init__(self, chunks: Iterable[DocumentChunk]) -> None:
        """Index unique chunks in stable identity order."""

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

        self._index = BM25Okapi(tokenized_chunks) if tokenized_chunks else None

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

        query_tokens = lexical_tokenize(query)

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
