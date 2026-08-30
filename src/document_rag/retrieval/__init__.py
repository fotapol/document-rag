"""Deterministic retrieval and evaluation services."""

from document_rag.retrieval.bm25 import (
    TABLE_QUERY_NORMALIZATION_STRATEGY,
    TOKENIZATION_STRATEGY,
    BM25Retriever,
    lexical_tokenize,
    normalize_bm25_query,
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
from document_rag.retrieval.hybrid import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_RRF_K,
    HybridRetrievalResult,
    ReciprocalRankFusionRetriever,
    fuse_rankings,
)
from document_rag.retrieval.models import (
    QueryRetrievalEvaluation,
    RetrievalMetrics,
    RetrievalResult,
)
from document_rag.retrieval.table_units import (
    ParentAwareRetriever,
    TableRetrievalCorpus,
    build_table_retrieval_corpus,
)

__all__ = [
    "DEFAULT_CANDIDATE_K",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_MODEL_REVISION",
    "DEFAULT_K_VALUES",
    "DEFAULT_RRF_K",
    "TABLE_QUERY_NORMALIZATION_STRATEGY",
    "TOKENIZATION_STRATEGY",
    "BM25Retriever",
    "DenseEmbedder",
    "DenseRetriever",
    "HybridRetrievalResult",
    "ParentAwareRetriever",
    "QueryRetrievalEvaluation",
    "ReciprocalRankFusionRetriever",
    "RetrievalEvaluationError",
    "RetrievalMetrics",
    "RetrievalResult",
    "SentenceTransformerEmbedder",
    "TableRetrievalCorpus",
    "aggregate_evaluations",
    "build_table_retrieval_corpus",
    "evaluate_query",
    "fuse_rankings",
    "lexical_tokenize",
    "normalize_bm25_query",
    "normalize_embedding_matrix",
    "normalize_k_values",
]
