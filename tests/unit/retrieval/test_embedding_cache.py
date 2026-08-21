"""Offline tests for deterministic document embedding caching."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from document_rag.ingestion.chunking import DocumentChunk
from document_rag.retrieval.dense import FloatMatrix
from document_rag.retrieval.embedding_cache import DocumentEmbeddingCache


class CountingEmbedder:
    """Track cache misses while returning controlled vectors."""

    def __init__(
        self,
        *,
        model_revision: str = "revision-1",
    ) -> None:
        self._model_revision = model_revision
        self.document_calls = 0

    @property
    def model_id(self) -> str:
        return "fake/model"

    @property
    def model_revision(self) -> str:
        return self._model_revision

    @property
    def dimension(self) -> int:
        return 2

    @property
    def device(self) -> str:
        return "cpu"

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> FloatMatrix:
        del batch_size
        self.document_calls += 1
        return np.asarray(
            [(float(index + 1), 1.0) for index, _ in enumerate(texts)],
            dtype=np.float32,
        )

    def embed_queries(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> FloatMatrix:
        del texts, batch_size
        raise AssertionError("Cache tests must not encode queries")


def build_chunks() -> tuple[DocumentChunk, ...]:
    """Build two stable cache rows."""

    return tuple(
        DocumentChunk(
            chunk_id=f"chunk-{index}",
            document_id="document",
            document_sha256="a" * 64,
            filename="report.pdf",
            chunk_index=index,
            page_start=1,
            page_end=1,
            source_element_ids=(f"source-{index}",),
            text=text,
            char_count=len(text),
            token_count=1,
            block_count=1,
        )
        for index, text in enumerate(("revenue", "operating income"))
    )


def test_valid_embedding_cache_is_reused(tmp_path: Path) -> None:
    """Matching model, corpus, and configuration should avoid re-encoding."""

    cache = DocumentEmbeddingCache(tmp_path / "cache")
    embedder = CountingEmbedder()
    chunks = build_chunks()

    first = cache.load_or_encode(
        chunks=chunks,
        embedder=embedder,
        batch_size=8,
        corpus_sha256="corpus-a",
    )
    second = cache.load_or_encode(
        chunks=chunks,
        embedder=embedder,
        batch_size=8,
        corpus_sha256="corpus-a",
    )

    assert first.reused is False
    assert second.reused is True
    assert embedder.document_calls == 1
    np.testing.assert_array_equal(first.embeddings, second.embeddings)
    assert first.embeddings.dtype == np.float32
    np.testing.assert_allclose(
        np.linalg.norm(first.embeddings, axis=1),
        np.ones(2),
    )


def test_model_revision_mismatch_invalidates_cache(tmp_path: Path) -> None:
    """Changing model provenance must recompute document vectors."""

    cache = DocumentEmbeddingCache(tmp_path / "cache")
    chunks = build_chunks()
    first_embedder = CountingEmbedder(model_revision="revision-1")
    second_embedder = CountingEmbedder(model_revision="revision-2")

    cache.load_or_encode(
        chunks=chunks,
        embedder=first_embedder,
        batch_size=8,
        corpus_sha256="corpus-a",
    )
    result = cache.load_or_encode(
        chunks=chunks,
        embedder=second_embedder,
        batch_size=8,
        corpus_sha256="corpus-a",
    )

    assert result.reused is False
    assert first_embedder.document_calls == 1
    assert second_embedder.document_calls == 1


def test_corpus_mismatch_invalidates_cache(tmp_path: Path) -> None:
    """Changing chunk content identity must recompute document vectors."""

    cache = DocumentEmbeddingCache(tmp_path / "cache")
    embedder = CountingEmbedder()
    chunks = build_chunks()

    cache.load_or_encode(
        chunks=chunks,
        embedder=embedder,
        batch_size=8,
        corpus_sha256="corpus-a",
    )
    result = cache.load_or_encode(
        chunks=chunks,
        embedder=embedder,
        batch_size=8,
        corpus_sha256="corpus-b",
    )

    assert result.reused is False
    assert embedder.document_calls == 2
