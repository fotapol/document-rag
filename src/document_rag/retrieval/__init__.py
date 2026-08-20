"""Deterministic retrieval and evaluation services."""

from document_rag.retrieval.bm25 import (
    TOKENIZATION_STRATEGY,
    BM25Retriever,
    lexical_tokenize,
)
from document_rag.retrieval.evaluation import (
    DEFAULT_K_VALUES,
    RetrievalEvaluationError,
    aggregate_evaluations,
    evaluate_query,
    normalize_k_values,
)
from document_rag.retrieval.models import (
    QueryRetrievalEvaluation,
    RetrievalMetrics,
    RetrievalResult,
)

__all__ = [
    "DEFAULT_K_VALUES",
    "TOKENIZATION_STRATEGY",
    "BM25Retriever",
    "QueryRetrievalEvaluation",
    "RetrievalEvaluationError",
    "RetrievalMetrics",
    "RetrievalResult",
    "aggregate_evaluations",
    "evaluate_query",
    "lexical_tokenize",
    "normalize_k_values",
]
