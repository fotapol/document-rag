"""Deterministic retrieval and evaluation services."""

from document_rag.retrieval.bm25 import (
    TOKENIZATION_STRATEGY,
    BM25Retriever,
    lexical_tokenize,
)
from document_rag.retrieval.dense import (
    DenseEmbedder,
    DenseRetriever,
    normalize_embedding_matrix,
)
from document_rag.retrieval.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_MODEL_REVISION,
    SentenceTransformerEmbedder,
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
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_MODEL_REVISION",
    "DEFAULT_K_VALUES",
    "TOKENIZATION_STRATEGY",
    "BM25Retriever",
    "DenseEmbedder",
    "DenseRetriever",
    "QueryRetrievalEvaluation",
    "RetrievalEvaluationError",
    "RetrievalMetrics",
    "RetrievalResult",
    "SentenceTransformerEmbedder",
    "aggregate_evaluations",
    "evaluate_query",
    "lexical_tokenize",
    "normalize_embedding_matrix",
    "normalize_k_values",
]
