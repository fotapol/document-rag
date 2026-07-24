"""Select DocFinQA chunks containing FinQA gold evidence."""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from document_rag.datasets.docfinqa.chunking import (
    DocFinQAChunk,
)
from document_rag.datasets.finqa.raw_models import (
    FinQARawRecord,
)

_NUMBER_COMMA_PATTERN = re.compile(r"(?<=\d),(?=\d)")
_NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9%.$-]+")
_TOKEN_PATTERN = re.compile(r"-?\$?\d+(?:\.\d+)?%?|[a-z]+")


@dataclass(frozen=True, slots=True)
class DocFinQAEvidenceMatch:
    """A FinQA supporting fact matched to one DocFinQA chunk."""

    source_key: str
    chunk_index: int
    score: float

    def __post_init__(self) -> None:
        if not self.source_key.strip():
            raise ValueError("Source key cannot be empty")

        if self.chunk_index < 0:
            raise ValueError("Chunk index cannot be negative")

        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Evidence score must be between 0 and 1")


class DocFinQAEvidenceSelector:
    """Match FinQA gold evidence against DocFinQA chunks.

    Exact normalized substring matching is attempted first.

    When formatting differs, the selector falls back to token
    coverage: the fraction of evidence tokens found in a chunk.

    A supporting fact is omitted when no chunk reaches the configured
    minimum score. The dataset preparer can then skip records without
    usable evidence.
    """

    def __init__(
        self,
        *,
        minimum_score: float = 0.6,
    ) -> None:
        if not 0.0 < minimum_score <= 1.0:
            raise ValueError("Minimum score must be greater than 0 and no greater than 1")

        self._minimum_score = minimum_score

    @property
    def minimum_score(self) -> float:
        return self._minimum_score

    def select(
        self,
        *,
        chunks: Iterable[DocFinQAChunk],
        finqa_record: FinQARawRecord,
    ) -> tuple[DocFinQAEvidenceMatch, ...]:
        """Return deterministic chunk matches for all gold facts."""

        ordered_chunks = tuple(
            sorted(
                chunks,
                key=lambda chunk: chunk.index,
            )
        )

        if not ordered_chunks:
            raise ValueError("At least one DocFinQA chunk is required")

        chunk_indexes = {chunk.index for chunk in ordered_chunks}

        if len(chunk_indexes) != len(ordered_chunks):
            raise ValueError("DocFinQA chunk indexes must be unique")

        matches: list[DocFinQAEvidenceMatch] = []

        for source_key, evidence_text in sorted(finqa_record.qa.gold_inds.items()):
            match = self._select_one(
                source_key=source_key,
                evidence_text=evidence_text,
                chunks=ordered_chunks,
            )

            if match is not None:
                matches.append(match)

        return tuple(matches)

    def _select_one(
        self,
        *,
        source_key: str,
        evidence_text: str,
        chunks: tuple[DocFinQAChunk, ...],
    ) -> DocFinQAEvidenceMatch | None:
        normalized_evidence = _normalize_text(evidence_text)

        if not normalized_evidence:
            return None

        evidence_tokens = _tokenize(normalized_evidence)

        if not evidence_tokens:
            return None

        best_chunk_index: int | None = None
        best_score = 0.0

        for chunk in chunks:
            normalized_chunk = _normalize_text(chunk.text)

            if normalized_evidence in normalized_chunk:
                score = 1.0
            else:
                chunk_tokens = _tokenize(normalized_chunk)
                score = _token_coverage(
                    evidence_tokens=evidence_tokens,
                    chunk_tokens=chunk_tokens,
                )

            if score > best_score:
                best_score = score
                best_chunk_index = chunk.index

        if best_chunk_index is None or best_score < self._minimum_score:
            return None

        return DocFinQAEvidenceMatch(
            source_key=source_key,
            chunk_index=best_chunk_index,
            score=best_score,
        )


def _normalize_text(value: str) -> str:
    normalized = value.casefold()
    normalized = _NUMBER_COMMA_PATTERN.sub(
        "",
        normalized,
    )
    normalized = _NON_ALPHANUMERIC_PATTERN.sub(
        " ",
        normalized,
    )

    return " ".join(normalized.split())


def _tokenize(value: str) -> frozenset[str]:
    return frozenset(_TOKEN_PATTERN.findall(value))


def _token_coverage(
    *,
    evidence_tokens: frozenset[str],
    chunk_tokens: frozenset[str],
) -> float:
    matching_tokens = evidence_tokens & chunk_tokens

    return len(matching_tokens) / len(evidence_tokens)
