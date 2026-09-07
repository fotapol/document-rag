"""Tests for environment-backed RAG configuration."""

import pytest

from document_rag.rag.config import (
    DEFAULT_ADAPTER_MODEL_ID,
    DEFAULT_ADAPTER_MODEL_REVISION,
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_BASE_MODEL_REVISION,
    RAGConfig,
)
from document_rag.rag.errors import RAGConfigurationError


def test_default_config_pins_base_model_and_adapter() -> None:
    """MVP defaults should select pinned base and published adapter revisions."""

    config = RAGConfig()

    assert config.top_k == 5
    assert config.max_table_rows_per_parent == 2
    assert config.base_model_id == DEFAULT_BASE_MODEL_ID
    assert config.base_model_revision == DEFAULT_BASE_MODEL_REVISION
    assert config.adapter_model_id == DEFAULT_ADAPTER_MODEL_ID
    assert config.adapter_model_revision == DEFAULT_ADAPTER_MODEL_REVISION
    assert config.adapter_model_id == "fotapol/qwen3-1.7b-financial-rag-lora-v4"
    assert config.adapter_model_revision == "d1af458b0bde4c74ccf87e3042ac8bb2390a8c65"


def test_environment_overrides_runtime_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deployment settings should be configurable without code changes."""

    monkeypatch.setenv("DOCUMENT_RAG_TOP_K", "3")
    monkeypatch.setenv("DOCUMENT_RAG_CANDIDATE_K", "12")
    monkeypatch.setenv("DOCUMENT_RAG_MAX_NEW_TOKENS", "64")
    monkeypatch.setenv("DOCUMENT_RAG_MAX_TABLE_ROWS_PER_PARENT", "1")
    monkeypatch.setenv("DOCUMENT_RAG_ADAPTER_MODEL_ID", "example/adapter")
    monkeypatch.setenv("DOCUMENT_RAG_ADAPTER_MODEL_REVISION", "adapter-sha")

    config = RAGConfig.from_environment()

    assert config.top_k == 3
    assert config.candidate_k == 12
    assert config.max_new_tokens == 64
    assert config.max_table_rows_per_parent == 1
    assert config.adapter_model_id == "example/adapter"
    assert config.adapter_model_revision == "adapter-sha"


def test_invalid_environment_integer_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configuration failures should be reported before model loading."""

    monkeypatch.setenv("DOCUMENT_RAG_TOP_K", "many")

    with pytest.raises(RAGConfigurationError, match="DOCUMENT_RAG_TOP_K must be an integer"):
        RAGConfig.from_environment()


def test_candidate_pool_must_cover_requested_results() -> None:
    """Each component candidate pool must be deep enough for the final top-K."""

    with pytest.raises(RAGConfigurationError, match="candidate_k"):
        RAGConfig(top_k=5, candidate_k=3)


def test_table_parent_limit_must_be_positive() -> None:
    """A disabled or negative diversity cap would make selection ambiguous."""

    with pytest.raises(RAGConfigurationError, match="max_table_rows_per_parent"):
        RAGConfig(max_table_rows_per_parent=0)
