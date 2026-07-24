"""Deterministic character-based chunking for DocFinQA reports."""

from collections.abc import Iterator
from dataclasses import dataclass

DEFAULT_CHUNK_SIZE = 2_750
DEFAULT_CHUNK_OVERLAP = 550


@dataclass(frozen=True, slots=True)
class DocFinQAChunk:
    """One text fragment with offsets into the original report."""

    index: int
    start_char: int
    end_char: int
    text: str

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("Chunk index cannot be negative")

        if self.start_char < 0:
            raise ValueError("Chunk start cannot be negative")

        if self.end_char <= self.start_char:
            raise ValueError("Chunk end must be greater than chunk start")

        if len(self.text) != self.end_char - self.start_char:
            raise ValueError("Chunk text length does not match its offsets")


class DocFinQAChunker:
    """Split large financial reports into overlapping text windows.

    The algorithm is intentionally character-based because DocFinQA
    benchmark contexts and the original retrieval procedure use
    character-sized windows.

    Source text is not stripped or normalized. Every chunk therefore
    remains an exact slice of the original report.
    """

    def __init__(
        self,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")

        if overlap < 0:
            raise ValueError("Chunk overlap cannot be negative")

        if overlap >= chunk_size:
            raise ValueError("Chunk overlap must be smaller than chunk size")

        self._chunk_size = chunk_size
        self._overlap = overlap
        self._stride = chunk_size - overlap

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def overlap(self) -> int:
        return self._overlap

    @property
    def stride(self) -> int:
        return self._stride

    def iter_chunks(
        self,
        context: str,
    ) -> Iterator[DocFinQAChunk]:
        """Yield exact source slices without loading extra documents."""

        if not context.strip():
            raise ValueError("Context cannot be empty")

        start_char = 0
        chunk_index = 0
        context_length = len(context)

        while start_char < context_length:
            end_char = min(
                start_char + self._chunk_size,
                context_length,
            )

            yield DocFinQAChunk(
                index=chunk_index,
                start_char=start_char,
                end_char=end_char,
                text=context[start_char:end_char],
            )

            if end_char == context_length:
                break

            start_char += self._stride
            chunk_index += 1
