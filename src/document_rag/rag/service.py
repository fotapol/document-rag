"""Application service that orchestrates in-memory retrieval and generation."""

from __future__ import annotations

from collections.abc import Iterable
from threading import RLock
from typing import Protocol

from document_rag.ingestion.chunking import DocumentChunk
from document_rag.rag.config import RAGConfig
from document_rag.rag.errors import RAGGenerationError, RAGIndexingError, RAGNotIndexedError
from document_rag.rag.generation import QwenLoraGenerator
from document_rag.rag.models import GroundedPrompt, RAGAnswer
from document_rag.rag.prompting import build_citations, build_grounded_prompt
from document_rag.retrieval.bm25 import BM25Retriever, normalize_bm25_query
from document_rag.retrieval.dense import DenseEmbedder, DenseRetriever
from document_rag.retrieval.diversity import ParentDiverseRetriever
from document_rag.retrieval.embeddings import SentenceTransformerEmbedder
from document_rag.retrieval.hybrid import RankedRetriever, ReciprocalRankFusionRetriever
from document_rag.retrieval.table_units import (
    ParentAwareRetriever,
    build_table_retrieval_corpus,
)


class RetrieverFactory(Protocol):
    """Build one in-memory ranked index from uploaded document chunks."""

    def build(self, chunks: tuple[DocumentChunk, ...]) -> RankedRetriever:
        """Return a ready-to-search retriever."""


class AnswerGenerator(Protocol):
    """Generate an answer from an already-grounded prompt."""

    def generate(self, prompt: GroundedPrompt) -> str:
        """Return model text for one prompt."""


class InMemoryHybridIndexFactory:
    """Compose the existing BM25, dense, and RRF implementations in memory."""

    def __init__(
        self,
        config: RAGConfig,
        *,
        embedder: DenseEmbedder | None = None,
    ) -> None:
        """Store retrieval configuration and an optional test embedder."""

        self._config = config
        self._embedder = embedder

    def build(self, chunks: tuple[DocumentChunk, ...]) -> ParentAwareRetriever:
        """Create fresh lexical and semantic indexes over one document."""

        if not chunks:
            raise RAGIndexingError("An uploaded document must contain at least one chunk.")

        try:
            embedder = self._get_embedder()
            corpus = build_table_retrieval_corpus(chunks)
            lexical_retriever = BM25Retriever(
                corpus.units,
                variant="plus",
                query_tokenizer=normalize_bm25_query,
            )
            semantic_retriever = DenseRetriever(
                corpus.units,
                embedder=embedder,
                batch_size=self._config.dense_batch_size,
            )
            fused_retriever = ReciprocalRankFusionRetriever(
                lexical_retriever=lexical_retriever,
                semantic_retriever=semantic_retriever,
                rrf_k=self._config.rrf_k,
                candidate_k=self._config.candidate_k,
            )
            return ParentAwareRetriever(
                retriever=ParentDiverseRetriever(
                    retriever=fused_retriever,
                    candidate_k=self._config.candidate_k,
                    max_table_rows_per_parent=self._config.max_table_rows_per_parent,
                ),
                corpus=corpus,
            )
        except RAGIndexingError:
            raise
        except Exception as exc:
            raise RAGIndexingError("Could not build the in-memory hybrid index.") from exc

    def _get_embedder(self) -> DenseEmbedder:
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder(
                model_id=self._config.embedding_model_id,
                model_revision=self._config.embedding_model_revision,
                device=self._config.embedding_device,
            )

        return self._embedder


class RAGService:
    """Own the current session index and execute grounded question answering."""

    def __init__(
        self,
        *,
        config: RAGConfig,
        retriever_factory: RetrieverFactory,
        generator: AnswerGenerator,
    ) -> None:
        """Inject retrieval and generation boundaries for offline testing."""

        self._config = config
        self._retriever_factory = retriever_factory
        self._generator = generator
        self._retriever: RankedRetriever | None = None
        self._chunks: tuple[DocumentChunk, ...] = ()
        self._lock = RLock()

    @property
    def is_indexed(self) -> bool:
        """Return whether this application session has an indexed document."""

        with self._lock:
            return self._retriever is not None

    @property
    def chunks(self) -> tuple[DocumentChunk, ...]:
        """Return the chunks associated with the current in-memory index."""

        with self._lock:
            return self._chunks

    def index_document(self, chunks: Iterable[DocumentChunk]) -> None:
        """Atomically replace the current index with one uploaded document."""

        materialized_chunks = tuple(chunks)
        retriever = self._retriever_factory.build(materialized_chunks)

        with self._lock:
            self._chunks = materialized_chunks
            self._retriever = retriever

    def answer(self, question: str) -> RAGAnswer:
        """Retrieve top-K chunks, build grounded context, and generate an answer."""

        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError("question must not be empty.")

        with self._lock:
            if self._retriever is None:
                raise RAGNotIndexedError("Upload and index a PDF before asking a question.")

            retrieved = self._retriever.search(
                normalized_question,
                top_k=self._config.top_k,
            )
            prompt = build_grounded_prompt(
                question=normalized_question,
                results=retrieved,
            )
            answer = self._generator.generate(prompt).strip()

        if not answer:
            raise RAGGenerationError("The answer model returned an empty response.")

        return RAGAnswer(
            question=normalized_question,
            answer=answer,
            citations=build_citations(retrieved),
            retrieved=retrieved,
        )


def build_default_rag_service(config: RAGConfig | None = None) -> RAGService:
    """Create the real application service with lazy local model adapters."""

    resolved_config = config or RAGConfig.from_environment()
    return RAGService(
        config=resolved_config,
        retriever_factory=InMemoryHybridIndexFactory(resolved_config),
        generator=QwenLoraGenerator(resolved_config),
    )
