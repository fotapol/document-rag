"""Tests for end-to-end RAG orchestration with offline fakes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pytest

from document_rag.ingestion.chunking import DocumentChunk
from document_rag.rag.config import RAGConfig
from document_rag.rag.errors import RAGGenerationError, RAGIndexingError, RAGNotIndexedError
from document_rag.rag.models import GroundedPrompt
from document_rag.rag.service import InMemoryHybridIndexFactory, RAGService
from document_rag.retrieval.dense import FloatMatrix
from document_rag.retrieval.hybrid import RankedRetriever
from document_rag.retrieval.models import RetrievalResult


@dataclass
class FakeRetriever:
    """Return pre-ranked chunks and record the requested top-K."""

    results: tuple[RetrievalResult, ...]
    searches: list[tuple[str, int]] = field(default_factory=list)

    def search(self, query: str, *, top_k: int = 5) -> tuple[RetrievalResult, ...]:
        self.searches.append((query, top_k))
        return self.results[:top_k]


@dataclass
class FakeRetrieverFactory:
    """Return one fake retriever and record indexed chunks."""

    retriever: RankedRetriever
    indexed_chunks: tuple[DocumentChunk, ...] = ()

    def build(self, chunks: tuple[DocumentChunk, ...]) -> RankedRetriever:
        self.indexed_chunks = chunks
        return self.retriever


@dataclass
class FakeGenerator:
    """Return deterministic text without loading a language model."""

    response: str
    prompts: list[GroundedPrompt] = field(default_factory=list)

    def generate(self, prompt: GroundedPrompt) -> str:
        self.prompts.append(prompt)
        return self.response


@dataclass(frozen=True)
class FakeEmbedder:
    """Provide tiny deterministic vectors without Internet or GPU access."""

    model_id: str = "fake/embedding"
    model_revision: str | None = "test"
    dimension: int = 2
    device: str = "cpu"

    def embed_documents(self, texts: Sequence[str], *, batch_size: int) -> FloatMatrix:
        assert batch_size > 0
        return np.asarray(
            [[1.0, 0.0] if "revenue" in text.casefold() else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )

    def embed_queries(self, texts: Sequence[str], *, batch_size: int) -> FloatMatrix:
        assert batch_size > 0
        return np.asarray(
            [[1.0, 0.0] if "revenue" in text.casefold() else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )


def test_service_indexes_retrieves_prompts_generates_and_cites() -> None:
    """The application service should execute the complete orchestration flow."""

    chunks = (_chunk("chunk:revenue", 3, "Revenue was $100."),)
    result = RetrievalResult(chunk=chunks[0], score=0.75, rank=1)
    retriever = FakeRetriever((result,))
    factory = FakeRetrieverFactory(retriever)
    generator = FakeGenerator("Revenue was $100 [Source 1].")
    service = RAGService(
        config=RAGConfig(top_k=1, candidate_k=1),
        retriever_factory=factory,
        generator=generator,
    )

    service.index_document(chunks)
    answer = service.answer("  What was revenue?  ")

    assert service.is_indexed is True
    assert service.chunks == chunks
    assert factory.indexed_chunks == chunks
    assert retriever.searches == [("What was revenue?", 1)]
    assert answer.question == "What was revenue?"
    assert answer.answer == "Revenue was $100 [Source 1]."
    assert answer.retrieved == (result,)
    assert answer.citations[0].chunk_id == "chunk:revenue"
    assert answer.citations[0].page_label == "3"
    assert "Revenue was $100." in generator.prompts[0].messages[1].content


def test_question_requires_an_index() -> None:
    """Questions asked before upload should fail with an actionable error."""

    service = RAGService(
        config=RAGConfig(top_k=1, candidate_k=1),
        retriever_factory=FakeRetrieverFactory(FakeRetriever(())),
        generator=FakeGenerator("unused"),
    )

    with pytest.raises(RAGNotIndexedError, match="Upload and index"):
        service.answer("What was revenue?")


def test_empty_generator_response_is_rejected() -> None:
    """The service should never render a silent model failure as an answer."""

    chunk = _chunk("chunk:revenue", 1, "Revenue was $100.")
    service = RAGService(
        config=RAGConfig(top_k=1, candidate_k=1),
        retriever_factory=FakeRetrieverFactory(
            FakeRetriever((RetrievalResult(chunk=chunk, score=1.0, rank=1),))
        ),
        generator=FakeGenerator("   "),
    )
    service.index_document((chunk,))

    with pytest.raises(RAGGenerationError, match="empty response"):
        service.answer("What was revenue?")


def test_real_in_memory_factory_reuses_bm25_dense_and_rrf() -> None:
    """A fake embedder should exercise the real existing retrieval composition."""

    chunks = (
        _chunk("chunk:revenue", 1, "Revenue increased to $100."),
        _chunk("chunk:employees", 2, "Employee count was 25."),
    )
    factory = InMemoryHybridIndexFactory(
        RAGConfig(top_k=2, candidate_k=2),
        embedder=FakeEmbedder(),
    )

    retriever = factory.build(chunks)
    first = retriever.search("revenue", top_k=2)
    second = retriever.search("revenue", top_k=2)

    assert first == second
    assert len(first) == 2
    assert {result.chunk_id for result in first} == {
        "chunk:employees",
        "chunk:revenue",
    }
    revenue_result = next(result for result in first if result.chunk_id == "chunk:revenue")
    assert revenue_result.chunk.source_element_ids == ("element:chunk:revenue",)


def test_real_factory_rejects_an_empty_document() -> None:
    """An empty upload must not replace a usable session index."""

    factory = InMemoryHybridIndexFactory(
        RAGConfig(top_k=1, candidate_k=1),
        embedder=FakeEmbedder(),
    )

    with pytest.raises(RAGIndexingError, match="at least one chunk"):
        factory.build(())


def test_real_factory_indexes_table_rows_instead_of_the_multirow_parent() -> None:
    """The RAG wiring should search canonical rows and retain parent recovery."""

    table = (
        "# Revenue\n\n<table><thead><tr><th>Quarter</th><th>Region</th>"
        "<th>Revenue</th></tr></thead><tbody>"
        "<tr><td>2024 Q1</td><td>Europe</td><td>$200</td></tr>"
        "<tr><td>2024 Q1</td><td>North America</td><td>$220</td></tr>"
        "</tbody></table>"
    )
    parent = _chunk("chunk:table", 3, table)
    factory = InMemoryHybridIndexFactory(
        RAGConfig(top_k=2, candidate_k=2),
        embedder=FakeEmbedder(),
    )

    retriever = factory.build((parent,))
    results = retriever.search("North America revenue 2024 Q1", top_k=2)

    assert len(results) == 2
    assert all(result.chunk.retrieval_unit_kind == "table_row" for result in results)
    assert all(result.chunk.chunk_id != parent.chunk_id for result in results)
    assert retriever.parent_for(results[0].chunk) == parent


def _chunk(chunk_id: str, page: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="document:test",
        document_sha256="test",
        filename="report.pdf",
        chunk_index=page - 1,
        page_start=page,
        page_end=page,
        source_element_ids=(f"element:{chunk_id}",),
        text=text,
        char_count=len(text),
        token_count=len(text.split()),
        block_count=1,
    )
