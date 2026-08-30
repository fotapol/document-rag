"""End-to-end retrieval-augmented question answering services."""

from document_rag.rag.config import (
    DEFAULT_ADAPTER_MODEL_ID,
    DEFAULT_ADAPTER_MODEL_REVISION,
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_BASE_MODEL_REVISION,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TOP_K,
    RAGConfig,
)
from document_rag.rag.errors import (
    RAGConfigurationError,
    RAGError,
    RAGGenerationError,
    RAGIndexingError,
    RAGNotIndexedError,
)
from document_rag.rag.generation import QwenLoraGenerator
from document_rag.rag.models import GroundedPrompt, RAGAnswer, SourceCitation
from document_rag.rag.prompting import SYSTEM_INSTRUCTION, UNSUPPORTED_ANSWER, build_grounded_prompt
from document_rag.rag.service import (
    InMemoryHybridIndexFactory,
    RAGService,
    build_default_rag_service,
)

__all__ = [
    "DEFAULT_ADAPTER_MODEL_ID",
    "DEFAULT_ADAPTER_MODEL_REVISION",
    "DEFAULT_BASE_MODEL_ID",
    "DEFAULT_BASE_MODEL_REVISION",
    "DEFAULT_MAX_INPUT_TOKENS",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_TOP_K",
    "SYSTEM_INSTRUCTION",
    "UNSUPPORTED_ANSWER",
    "GroundedPrompt",
    "InMemoryHybridIndexFactory",
    "QwenLoraGenerator",
    "RAGAnswer",
    "RAGConfig",
    "RAGConfigurationError",
    "RAGError",
    "RAGGenerationError",
    "RAGIndexingError",
    "RAGNotIndexedError",
    "RAGService",
    "SourceCitation",
    "build_default_rag_service",
    "build_grounded_prompt",
]
