from itertools import pairwise

import pytest

from document_rag.datasets.docfinqa.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DocFinQAChunk,
    DocFinQAChunker,
)


def test_chunker_uses_benchmark_defaults() -> None:
    chunker = DocFinQAChunker()

    assert chunker.chunk_size == 2_750
    assert chunker.overlap == 550
    assert chunker.stride == 2_200

    assert chunker.chunk_size == DEFAULT_CHUNK_SIZE
    assert chunker.overlap == DEFAULT_CHUNK_OVERLAP


def test_chunker_returns_one_chunk_for_short_context() -> None:
    context = "Annual report content."
    chunker = DocFinQAChunker()

    chunks = list(chunker.iter_chunks(context))

    assert chunks == [
        DocFinQAChunk(
            index=0,
            start_char=0,
            end_char=len(context),
            text=context,
        )
    ]


def test_chunker_returns_one_chunk_at_exact_size() -> None:
    context = "x" * 2_750
    chunker = DocFinQAChunker()

    chunks = list(chunker.iter_chunks(context))

    assert len(chunks) == 1
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == 2_750
    assert chunks[0].text == context


def test_chunker_creates_overlapping_chunks() -> None:
    context = "".join(str(index % 10) for index in range(6_000))
    chunker = DocFinQAChunker()

    chunks = list(chunker.iter_chunks(context))

    assert len(chunks) == 3

    assert chunks[0].index == 0
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == 2_750

    assert chunks[1].index == 1
    assert chunks[1].start_char == 2_200
    assert chunks[1].end_char == 4_950

    assert chunks[2].index == 2
    assert chunks[2].start_char == 4_400
    assert chunks[2].end_char == 6_000


def test_chunks_are_exact_source_slices() -> None:
    context = "".join(chr(65 + index % 26) for index in range(8_000))
    chunker = DocFinQAChunker()

    chunks = list(chunker.iter_chunks(context))

    for chunk in chunks:
        assert chunk.text == context[chunk.start_char : chunk.end_char]


def test_adjacent_chunks_have_expected_overlap() -> None:
    context = "x" * 8_000
    chunker = DocFinQAChunker(
        chunk_size=1_000,
        overlap=200,
    )

    chunks = list(chunker.iter_chunks(context))

    for previous, current in pairwise(chunks):
        assert previous.end_char - current.start_char == 200

        assert previous.text[-200:] == current.text[:200]


def test_chunking_is_deterministic() -> None:
    context = "Financial report. " * 1_000
    chunker = DocFinQAChunker()

    first_result = list(chunker.iter_chunks(context))
    second_result = list(chunker.iter_chunks(context))

    assert first_result == second_result


def test_chunker_supports_zero_overlap() -> None:
    context = "abcdefghij"
    chunker = DocFinQAChunker(
        chunk_size=4,
        overlap=0,
    )

    chunks = list(chunker.iter_chunks(context))

    assert [chunk.text for chunk in chunks] == [
        "abcd",
        "efgh",
        "ij",
    ]

    assert [chunk.start_char for chunk in chunks] == [
        0,
        4,
        8,
    ]


@pytest.mark.parametrize(
    "chunk_size",
    [
        0,
        -1,
    ],
)
def test_chunker_rejects_invalid_chunk_size(
    chunk_size: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="Chunk size must be positive",
    ):
        DocFinQAChunker(
            chunk_size=chunk_size,
        )


@pytest.mark.parametrize(
    "overlap",
    [
        -1,
        1_000,
        1_001,
    ],
)
def test_chunker_rejects_invalid_overlap(
    overlap: int,
) -> None:
    with pytest.raises(ValueError):
        DocFinQAChunker(
            chunk_size=1_000,
            overlap=overlap,
        )


@pytest.mark.parametrize(
    "context",
    [
        "",
        " ",
        "\n\t",
    ],
)
def test_chunker_rejects_empty_context(
    context: str,
) -> None:
    chunker = DocFinQAChunker()

    with pytest.raises(
        ValueError,
        match="Context cannot be empty",
    ):
        list(chunker.iter_chunks(context))


def test_chunk_rejects_inconsistent_offsets() -> None:
    with pytest.raises(
        ValueError,
        match="text length does not match",
    ):
        DocFinQAChunk(
            index=0,
            start_char=10,
            end_char=20,
            text="short",
        )
