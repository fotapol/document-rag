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
    RAGEvaluationError,
    RAGGenerationError,
    RAGIndexingError,
    RAGNotIndexedError,
)
from document_rag.rag.evaluation import (
    RAG_EVALUATION_SCHEMA_VERSION,
    AnswerAssessment,
    FrozenRAGCase,
    FrozenRAGSuite,
    ModelEvaluationMetrics,
    RAGCaseExpectation,
    RAGEvaluationResult,
    assess_answer,
    load_frozen_rag_suite,
    run_frozen_rag_evaluation,
)
from document_rag.rag.generation import (
    QwenBaseLoraComparisonGenerator,
    QwenLoraGenerator,
)
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
    "RAG_EVALUATION_SCHEMA_VERSION",
    "SYSTEM_INSTRUCTION",
    "UNSUPPORTED_ANSWER",
    "AnswerAssessment",
    "FrozenRAGCase",
    "FrozenRAGSuite",
    "GroundedPrompt",
    "InMemoryHybridIndexFactory",
    "ModelEvaluationMetrics",
    "QwenBaseLoraComparisonGenerator",
    "QwenLoraGenerator",
    "RAGAnswer",
    "RAGCaseExpectation",
    "RAGConfig",
    "RAGConfigurationError",
    "RAGError",
    "RAGEvaluationError",
    "RAGEvaluationResult",
    "RAGGenerationError",
    "RAGIndexingError",
    "RAGNotIndexedError",
    "RAGService",
    "SourceCitation",
    "assess_answer",
    "build_default_rag_service",
    "build_grounded_prompt",
    "load_frozen_rag_suite",
    "run_frozen_rag_evaluation",
]
