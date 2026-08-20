"""Immutable retrieval results and metric records."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from document_rag.datasets.models import DatasetName
from document_rag.ingestion.chunking import DocumentChunk


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """One ranked document chunk and its retriever score."""

    chunk: DocumentChunk
    score: float
    rank: int

    def __post_init__(self) -> None:
        """Reject invalid rankings before they reach evaluation artifacts."""

        if self.rank <= 0:
            raise ValueError("rank must be positive.")

        if not isfinite(self.score):
            raise ValueError("score must be finite.")

    @property
    def chunk_id(self) -> str:
        """Return the stable ID of the retrieved chunk."""

        return self.chunk.chunk_id

    def to_record(self) -> dict[str, object]:
        """Return the debugging fields stored for one retrieved chunk."""

        return {
            "chunk_id": self.chunk.chunk_id,
            "document_id": self.chunk.document_id,
            "page_end": self.chunk.page_end,
            "page_start": self.chunk.page_start,
            "rank": self.rank,
            "score": self.score,
            "source_element_ids": list(self.chunk.source_element_ids),
        }


@dataclass(frozen=True, slots=True)
class QueryRetrievalEvaluation:
    """Serializable retrieval outcome and metrics for one question."""

    dataset: DatasetName
    example_id: str
    question: str
    gold_source_element_ids: tuple[str, ...]
    results: tuple[RetrievalResult, ...]
    first_relevant_rank: int | None
    hit_at_k: tuple[tuple[int, int], ...]
    recall_at_k: tuple[tuple[int, float], ...]

    def to_record(self) -> dict[str, object]:
        """Convert the query evaluation to a deterministic JSON record."""

        return {
            "dataset": self.dataset.value,
            "example_id": self.example_id,
            "first_relevant_rank": self.first_relevant_rank,
            "gold_source_element_ids": list(self.gold_source_element_ids),
            "hit_at_k": {str(k): value for k, value in self.hit_at_k},
            "question": self.question,
            "recall_at_k": {str(k): value for k, value in self.recall_at_k},
            "retrieved": [result.to_record() for result in self.results],
        }


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Macro-averaged retrieval metrics across evaluable questions."""

    query_count: int
    hit_rate_at_k: tuple[tuple[int, float], ...]
    recall_at_k: tuple[tuple[int, float], ...]
    mrr: float

    def to_record(self) -> dict[str, object]:
        """Convert aggregate metrics to a deterministic JSON record."""

        return {
            "hit_rate_at_k": {str(k): value for k, value in self.hit_rate_at_k},
            "mrr": self.mrr,
            "query_count": self.query_count,
            "recall_at_k": {str(k): value for k, value in self.recall_at_k},
        }
