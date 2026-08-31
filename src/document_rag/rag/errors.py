"""Errors raised by the end-to-end RAG application layer."""


class RAGError(RuntimeError):
    """Base error for user-facing RAG failures."""


class RAGConfigurationError(RAGError):
    """Raised when RAG environment configuration is invalid."""


class RAGIndexingError(RAGError):
    """Raised when an uploaded document cannot be indexed."""


class RAGNotIndexedError(RAGError):
    """Raised when a question is asked before a document is indexed."""


class RAGGenerationError(RAGError):
    """Raised when the answer model cannot load or generate a response."""


class RAGEvaluationError(RAGError):
    """Raised when a frozen RAG evaluation suite or run is invalid."""
