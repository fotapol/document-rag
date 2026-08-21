"""Sentence Transformers embedding adapter for the dense baseline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from document_rag.retrieval.dense import FloatMatrix

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BGE_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "
DOCUMENT_ENCODING_STRATEGY = "sentence_transformers_encode_document_v1"
QUERY_ENCODING_STRATEGY = "bge_query_instruction_v1"


class SentenceTransformerEmbedder:
    """Encode BGE documents and instructed queries without import-time downloads."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_EMBEDDING_MODEL,
        model_revision: str | None = DEFAULT_EMBEDDING_MODEL_REVISION,
        device: str = "cpu",
        show_progress: bool = False,
    ) -> None:
        """Load one pinned Sentence Transformers model on the selected device."""

        if not model_id.strip():
            raise ValueError("model_id must not be empty.")

        if model_revision is not None and not model_revision.strip():
            raise ValueError("model_revision must be non-empty when provided.")

        if not device.strip():
            raise ValueError("device must not be empty.")

        from sentence_transformers import SentenceTransformer

        self._model_id = model_id
        self._model_revision = model_revision
        self._model: Any = SentenceTransformer(
            model_id,
            revision=model_revision,
            device=None if device == "auto" else device,
        )
        dimension = self._model.get_embedding_dimension()

        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Embedding model did not expose a valid dimension.")

        self._dimension = dimension
        self._device = str(self._model.device)
        self._show_progress = show_progress

    @property
    def model_id(self) -> str:
        """Return the Hugging Face model identifier."""

        return self._model_id

    @property
    def model_revision(self) -> str | None:
        """Return the requested immutable Hugging Face revision."""

        return self._model_revision

    @property
    def dimension(self) -> int:
        """Return the model's sentence-embedding dimension."""

        return self._dimension

    @property
    def device(self) -> str:
        """Return the actual Sentence Transformers inference device."""

        return self._device

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> FloatMatrix:
        """Encode raw chunk text in deterministic input order."""

        return self._encode(
            method_name="encode_document",
            texts=texts,
            batch_size=batch_size,
        )

    def embed_queries(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> FloatMatrix:
        """Encode only question text with the documented BGE query prompt."""

        return self._encode(
            method_name="encode_query",
            texts=texts,
            batch_size=batch_size,
            prompt=BGE_QUERY_PROMPT,
        )

    def _encode(
        self,
        *,
        method_name: str,
        texts: Sequence[str],
        batch_size: int,
        prompt: str | None = None,
    ) -> FloatMatrix:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        method = getattr(self._model, method_name)
        values = method(
            list(texts),
            prompt=prompt,
            batch_size=batch_size,
            show_progress_bar=self._show_progress,
            precision="float32",
            convert_to_numpy=True,
            convert_to_tensor=False,
            normalize_embeddings=False,
        )
        return np.asarray(values, dtype=np.float32)
