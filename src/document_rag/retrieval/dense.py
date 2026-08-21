"""Exact deterministic dense retrieval over normalized document chunks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from document_rag.ingestion.chunking import DocumentChunk
from document_rag.retrieval.models import RetrievalResult

type FloatMatrix = NDArray[np.float32]
type FloatVector = NDArray[np.float32]


class DenseEmbedder(Protocol):
    """Minimal document/query embedding interface used by dense retrieval."""

    @property
    def model_id(self) -> str:
        """Return the embedding model identifier."""

    @property
    def model_revision(self) -> str | None:
        """Return the pinned model revision when known."""

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""

    @property
    def device(self) -> str:
        """Return the inference device actually in use."""

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> FloatMatrix:
        """Encode document texts as a two-dimensional numeric array."""

    def embed_queries(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> FloatMatrix:
        """Encode retrieval queries as a two-dimensional numeric array."""


def normalize_embedding_matrix(
    embeddings: object,
    *,
    expected_rows: int,
    expected_dimension: int | None = None,
    label: str,
) -> FloatMatrix:
    """Validate and L2-normalize a float32 embedding matrix by row."""

    matrix = np.asarray(embeddings, dtype=np.float32)

    if matrix.ndim != 2:
        raise ValueError(f"{label} embeddings must be a two-dimensional matrix.")

    if matrix.shape[0] != expected_rows:
        raise ValueError(f"{label} embedding row count must match its input count.")

    if matrix.shape[1] <= 0:
        raise ValueError(f"{label} embeddings must have a positive dimension.")

    if expected_dimension is not None and matrix.shape[1] != expected_dimension:
        raise ValueError(
            f"{label} embedding dimension must be {expected_dimension}, got {matrix.shape[1]}."
        )

    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} embeddings must contain only finite values.")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)

    if np.any(norms == 0):
        raise ValueError(f"{label} embeddings cannot contain zero vectors.")

    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


class DenseRetriever:
    """Exact cosine retriever using normalized float32 NumPy vectors."""

    def __init__(
        self,
        chunks: Iterable[DocumentChunk],
        *,
        embedder: DenseEmbedder,
        batch_size: int = 32,
        document_embeddings: object | None = None,
    ) -> None:
        """Build a stable chunk index or accept matching precomputed vectors.

        Precomputed rows must follow the retriever's stable chunk order:
        document ID, chunk index, then chunk ID.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        materialized_chunks = tuple(chunks)
        chunk_ids = [chunk.chunk_id for chunk in materialized_chunks]

        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Dense chunks must have unique chunk IDs.")

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
        self._embedder = embedder
        self._batch_size = batch_size

        if not self._chunks:
            if document_embeddings is not None:
                matrix = np.asarray(document_embeddings)

                if matrix.shape[0] != 0:
                    raise ValueError("Empty dense indexes cannot contain document embeddings.")

            self._document_embeddings = np.empty(
                (0, embedder.dimension),
                dtype=np.float32,
            )
            return

        raw_embeddings = (
            embedder.embed_documents(
                tuple(chunk.text for chunk in self._chunks),
                batch_size=batch_size,
            )
            if document_embeddings is None
            else document_embeddings
        )
        self._document_embeddings = normalize_embedding_matrix(
            raw_embeddings,
            expected_rows=len(self._chunks),
            expected_dimension=embedder.dimension,
            label="Document",
        )

    @property
    def chunk_count(self) -> int:
        """Return the number of indexed chunks."""

        return len(self._chunks)

    @property
    def document_embeddings(self) -> FloatMatrix:
        """Return a defensive copy of normalized document vectors."""

        return self._document_embeddings.copy()

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> tuple[RetrievalResult, ...]:
        """Embed one question and return its exact cosine ranking."""

        self._validate_search(query=query, top_k=top_k)

        if not self._chunks:
            return ()

        query_embeddings = self._embedder.embed_queries(
            (query,),
            batch_size=self._batch_size,
        )
        normalized_query = normalize_embedding_matrix(
            query_embeddings,
            expected_rows=1,
            expected_dimension=self._document_embeddings.shape[1],
            label="Query",
        )[0]
        return self._search_normalized_vector(normalized_query, top_k=top_k)

    def search_vector(
        self,
        query_embedding: object,
        *,
        top_k: int = 5,
    ) -> tuple[RetrievalResult, ...]:
        """Rank chunks from one precomputed query embedding."""

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if not self._chunks:
            return ()

        vector = np.asarray(query_embedding, dtype=np.float32)

        if vector.ndim != 1:
            raise ValueError("A query embedding must be one-dimensional.")

        normalized_query = normalize_embedding_matrix(
            vector[np.newaxis, :],
            expected_rows=1,
            expected_dimension=self._document_embeddings.shape[1],
            label="Query",
        )[0]
        return self._search_normalized_vector(normalized_query, top_k=top_k)

    def _validate_search(self, *, query: str, top_k: int) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if not query.strip():
            raise ValueError("query must not be empty.")

    def _search_normalized_vector(
        self,
        query_embedding: FloatVector,
        *,
        top_k: int,
    ) -> tuple[RetrievalResult, ...]:
        scores = self._document_embeddings @ query_embedding
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
