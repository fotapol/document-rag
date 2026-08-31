"""Environment-backed configuration for the RAG MVP."""

from __future__ import annotations

import os
from dataclasses import dataclass

from document_rag.rag.errors import RAGConfigurationError
from document_rag.retrieval.diversity import DEFAULT_MAX_TABLE_ROWS_PER_PARENT
from document_rag.retrieval.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_MODEL_REVISION,
)
from document_rag.retrieval.hybrid import DEFAULT_CANDIDATE_K, DEFAULT_RRF_K

DEFAULT_BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
DEFAULT_BASE_MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DEFAULT_ADAPTER_MODEL_ID = "fotapol/qwen3-1.7b-financial-qa-lora"
DEFAULT_ADAPTER_MODEL_REVISION = "8433fcf8142e3db64f38df1f6eeaa38bf2e92651"
DEFAULT_TOP_K = 5
DEFAULT_DENSE_BATCH_SIZE = 32
DEFAULT_MAX_INPUT_TOKENS = 4096
DEFAULT_MAX_NEW_TOKENS = 128


@dataclass(frozen=True, slots=True)
class RAGConfig:
    """Runtime configuration for retrieval and deterministic generation."""

    top_k: int = DEFAULT_TOP_K
    candidate_k: int = DEFAULT_CANDIDATE_K
    rrf_k: int = DEFAULT_RRF_K
    max_table_rows_per_parent: int = DEFAULT_MAX_TABLE_ROWS_PER_PARENT
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL
    embedding_model_revision: str | None = DEFAULT_EMBEDDING_MODEL_REVISION
    embedding_device: str = "auto"
    dense_batch_size: int = DEFAULT_DENSE_BATCH_SIZE
    base_model_id: str = DEFAULT_BASE_MODEL_ID
    base_model_revision: str = DEFAULT_BASE_MODEL_REVISION
    adapter_model_id: str = DEFAULT_ADAPTER_MODEL_ID
    adapter_model_revision: str | None = DEFAULT_ADAPTER_MODEL_REVISION
    generation_device_map: str = "auto"
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS

    def __post_init__(self) -> None:
        """Reject invalid configuration before loading any models."""

        positive_values = {
            "top_k": self.top_k,
            "candidate_k": self.candidate_k,
            "rrf_k": self.rrf_k,
            "max_table_rows_per_parent": self.max_table_rows_per_parent,
            "dense_batch_size": self.dense_batch_size,
            "max_input_tokens": self.max_input_tokens,
            "max_new_tokens": self.max_new_tokens,
        }

        for setting_name, number in positive_values.items():
            if number <= 0:
                raise RAGConfigurationError(f"{setting_name} must be positive.")

        if self.candidate_k < self.top_k:
            raise RAGConfigurationError("candidate_k must be greater than or equal to top_k.")

        required_strings = {
            "embedding_model_id": self.embedding_model_id,
            "embedding_device": self.embedding_device,
            "base_model_id": self.base_model_id,
            "base_model_revision": self.base_model_revision,
            "adapter_model_id": self.adapter_model_id,
            "generation_device_map": self.generation_device_map,
        }

        for setting_name, text in required_strings.items():
            if not text.strip():
                raise RAGConfigurationError(f"{setting_name} must not be empty.")

        optional_revisions = {
            "embedding_model_revision": self.embedding_model_revision,
            "adapter_model_revision": self.adapter_model_revision,
        }

        for setting_name, revision in optional_revisions.items():
            if revision is not None and not revision.strip():
                raise RAGConfigurationError(f"{setting_name} must be non-empty when provided.")

    @classmethod
    def from_environment(cls) -> RAGConfig:
        """Read supported overrides from ``DOCUMENT_RAG_*`` variables."""

        return cls(
            top_k=_environment_int("DOCUMENT_RAG_TOP_K", DEFAULT_TOP_K),
            candidate_k=_environment_int(
                "DOCUMENT_RAG_CANDIDATE_K",
                DEFAULT_CANDIDATE_K,
            ),
            rrf_k=_environment_int("DOCUMENT_RAG_RRF_K", DEFAULT_RRF_K),
            max_table_rows_per_parent=_environment_int(
                "DOCUMENT_RAG_MAX_TABLE_ROWS_PER_PARENT",
                DEFAULT_MAX_TABLE_ROWS_PER_PARENT,
            ),
            embedding_model_id=os.getenv(
                "DOCUMENT_RAG_EMBEDDING_MODEL_ID",
                DEFAULT_EMBEDDING_MODEL,
            ),
            embedding_model_revision=_optional_environment_value(
                "DOCUMENT_RAG_EMBEDDING_MODEL_REVISION",
                DEFAULT_EMBEDDING_MODEL_REVISION,
            ),
            embedding_device=os.getenv("DOCUMENT_RAG_EMBEDDING_DEVICE", "auto"),
            dense_batch_size=_environment_int(
                "DOCUMENT_RAG_DENSE_BATCH_SIZE",
                DEFAULT_DENSE_BATCH_SIZE,
            ),
            base_model_id=os.getenv(
                "DOCUMENT_RAG_BASE_MODEL_ID",
                DEFAULT_BASE_MODEL_ID,
            ),
            base_model_revision=os.getenv(
                "DOCUMENT_RAG_BASE_MODEL_REVISION",
                DEFAULT_BASE_MODEL_REVISION,
            ),
            adapter_model_id=os.getenv(
                "DOCUMENT_RAG_ADAPTER_MODEL_ID",
                DEFAULT_ADAPTER_MODEL_ID,
            ),
            adapter_model_revision=_optional_environment_value(
                "DOCUMENT_RAG_ADAPTER_MODEL_REVISION",
                DEFAULT_ADAPTER_MODEL_REVISION,
            ),
            generation_device_map=os.getenv(
                "DOCUMENT_RAG_GENERATION_DEVICE_MAP",
                "auto",
            ),
            max_input_tokens=_environment_int(
                "DOCUMENT_RAG_MAX_INPUT_TOKENS",
                DEFAULT_MAX_INPUT_TOKENS,
            ),
            max_new_tokens=_environment_int(
                "DOCUMENT_RAG_MAX_NEW_TOKENS",
                DEFAULT_MAX_NEW_TOKENS,
            ),
        )


def _environment_int(name: str, default: int) -> int:
    value = os.getenv(name)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise RAGConfigurationError(f"{name} must be an integer.") from exc


def _optional_environment_value(name: str, default: str | None) -> str | None:
    value = os.getenv(name)

    if value is None:
        return default

    stripped = value.strip()
    return stripped or None
