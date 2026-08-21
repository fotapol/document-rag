"""Offline tests for exact deterministic dense retrieval."""

from collections.abc import Sequence

import numpy as np
import pytest

from document_rag.datasets.models import DatasetName
from document_rag.ingestion.chunking import DocumentChunk
from document_rag.retrieval.dense import DenseRetriever, FloatMatrix
from document_rag.retrieval.evaluation import evaluate_query
from document_rag.retrieval.models import RetrievalResult


class FakeEmbedder:
    """Return controlled vectors without model downloads or GPU access."""

    def __init__(
        self,
        vectors: dict[str, tuple[float, ...]],
        *,
        model_id: str = "fake/model",
        model_revision: str = "revision-1",
    ) -> None:
        self.vectors = vectors
        self._model_id = model_id
        self._model_revision = model_revision
        self.document_calls = 0
        self.query_calls = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_revision(self) -> str:
        return self._model_revision

    @property
    def dimension(self) -> int:
        return len(next(iter(self.vectors.values())))

    @property
    def device(self) -> str:
        return "cpu"

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> FloatMatrix:
        assert batch_size > 0
        self.document_calls += 1
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)

    def embed_queries(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> FloatMatrix:
        assert batch_size > 0
        self.query_calls += 1
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


class InvalidShapeEmbedder(FakeEmbedder):
    """Return one configured invalid document embedding payload."""

    def __init__(self, payload: object) -> None:
        super().__init__({"query": (1.0, 0.0), "chunk": (1.0, 0.0)})
        self.payload = payload

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> FloatMatrix:
        del texts, batch_size
        return np.asarray(self.payload, dtype=np.float32)


def build_chunk(
    chunk_id: str,
    chunk_index: int,
    text: str,
    source_id: str,
) -> DocumentChunk:
    """Build one immutable synthetic retrieval chunk."""

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="document",
        document_sha256="a" * 64,
        filename="report.pdf",
        chunk_index=chunk_index,
        page_start=chunk_index + 1,
        page_end=chunk_index + 1,
        source_element_ids=(source_id,),
        text=text,
        char_count=len(text),
        token_count=1,
        block_count=1,
    )


def build_retriever() -> DenseRetriever:
    """Build a controlled semantic ranking fixture."""

    embedder = FakeEmbedder(
        {
            "cash": (1.0, 0.0),
            "operating income": (0.0, 3.0),
            "compensation": (-1.0, 0.0),
            "income question": (0.0, 7.0),
        }
    )
    return DenseRetriever(
        (
            build_chunk("chunk-cash", 0, "cash", "cash-source"),
            build_chunk(
                "chunk-income",
                1,
                "operating income",
                "income-source",
            ),
            build_chunk(
                "chunk-compensation",
                2,
                "compensation",
                "compensation-source",
            ),
        ),
        embedder=embedder,
    )


def test_semantic_query_ranks_intended_chunk_and_preserves_lineage() -> None:
    """Controlled semantic similarity should rank the gold chunk first."""

    results = build_retriever().search("income question", top_k=3)

    assert isinstance(results[0], RetrievalResult)
    assert results[0].chunk_id == "chunk-income"
    assert results[0].chunk.source_element_ids == ("income-source",)
    assert results[0].score == pytest.approx(1.0)
    assert results[0].score > results[1].score


def test_top_k_is_respected_without_duplicate_chunks() -> None:
    """Dense ranking should be unique and bounded by the requested cutoff."""

    results = build_retriever().search("income question", top_k=2)

    assert len(results) == 2
    assert [result.rank for result in results] == [1, 2]
    assert len({result.chunk_id for result in results}) == 2


def test_top_k_greater_than_corpus_returns_every_chunk() -> None:
    """A large cutoff should not pad or duplicate results."""

    results = build_retriever().search("income question", top_k=10)

    assert len(results) == 3
    assert [result.rank for result in results] == [1, 2, 3]


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_is_rejected(top_k: int) -> None:
    """Non-positive dense cutoffs are invalid."""

    with pytest.raises(ValueError, match="top_k must be positive"):
        build_retriever().search("income question", top_k=top_k)


def test_document_and_query_vectors_are_l2_normalized() -> None:
    """Cosine ranking should normalize both sides before dot products."""

    embedder = FakeEmbedder(
        {
            "horizontal": (3.0, 0.0),
            "vertical": (0.0, 4.0),
            "query": (3.0, 4.0),
        }
    )
    retriever = DenseRetriever(
        (
            build_chunk("chunk-horizontal", 0, "horizontal", "horizontal"),
            build_chunk("chunk-vertical", 1, "vertical", "vertical"),
        ),
        embedder=embedder,
    )

    np.testing.assert_allclose(
        retriever.document_embeddings,
        np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32),
    )
    results = retriever.search("query", top_k=2)
    assert [result.chunk_id for result in results] == [
        "chunk-vertical",
        "chunk-horizontal",
    ]
    assert [result.score for result in results] == pytest.approx([0.8, 0.6])


def test_equal_scores_use_chunk_id_as_deterministic_tie_breaker() -> None:
    """Equal cosine scores should sort by stable chunk identity."""

    embedder = FakeEmbedder(
        {
            "first": (1.0, 0.0),
            "second": (1.0, 0.0),
            "query": (1.0, 0.0),
        }
    )
    chunks = (
        build_chunk("chunk-z", 0, "first", "first"),
        build_chunk("chunk-a", 1, "second", "second"),
    )

    results = DenseRetriever(chunks, embedder=embedder).search("query", top_k=2)

    assert [result.chunk_id for result in results] == ["chunk-a", "chunk-z"]


def test_identical_inputs_produce_identical_ranking() -> None:
    """Repeated dense construction and search should be deterministic."""

    first = build_retriever().search("income question", top_k=3)
    second = build_retriever().search("income question", top_k=3)

    assert first == second


def test_empty_corpus_returns_no_results_without_encoding() -> None:
    """An empty dense index should be explicit and avoid model inference."""

    embedder = FakeEmbedder({"query": (1.0, 0.0)})
    retriever = DenseRetriever((), embedder=embedder)

    assert retriever.chunk_count == 0
    assert retriever.search("query", top_k=5) == ()
    assert embedder.document_calls == 0
    assert embedder.query_calls == 0


@pytest.mark.parametrize(
    "payload",
    [
        [1.0, 0.0],
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0, 0.0]],
        [[0.0, 0.0]],
        [[float("nan"), 1.0]],
    ],
)
def test_invalid_document_embedding_shapes_and_values_are_rejected(
    payload: object,
) -> None:
    """Malformed model output must fail before ranking."""

    with pytest.raises(ValueError, match=r"Document embeddings|Document embedding"):
        DenseRetriever(
            (build_chunk("chunk", 0, "chunk", "source"),),
            embedder=InvalidShapeEmbedder(payload),
        )


def test_existing_evaluator_consumes_dense_results_unchanged() -> None:
    """Dense results should use the same source-lineage metric implementation."""

    results = build_retriever().search("income question", top_k=3)
    evaluation = evaluate_query(
        dataset=DatasetName.FINQA,
        example_id="dense-example",
        question="income question",
        gold_source_element_ids=("income-source",),
        results=results,
    )

    assert evaluation.hit_at_k == ((1, 1), (3, 1), (5, 1))
    assert evaluation.recall_at_k == ((1, 1.0), (3, 1.0), (5, 1.0))
    assert evaluation.first_relevant_rank == 1
