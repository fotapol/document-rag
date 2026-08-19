"""Map gold source-element annotations to retrieval chunks."""

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from document_rag.datasets.models import SupportingFact
from document_rag.domain.documents import DocumentElement
from document_rag.ingestion.chunking import DocumentChunk

_NON_EVIDENCE_TOKEN_PATTERN = re.compile(r"[^\w%.$-]+", re.UNICODE)


class GoldEvidenceMappingError(ValueError):
    """Raised when gold evidence cannot be mapped completely and unambiguously."""


@dataclass(frozen=True, slots=True)
class GoldEvidenceChunkMapping:
    """All retrieval chunks that retain one gold source element."""

    source_key: str
    source_element_id: str
    lineage_element_id: str
    chunk_ids: tuple[str, ...]
    chunk_indexes: tuple[int, ...]

    def __post_init__(self) -> None:
        """Reject incomplete or internally inconsistent mappings."""

        if not self.source_key.strip():
            raise ValueError("source_key cannot be empty.")

        if not self.source_element_id.strip():
            raise ValueError("source_element_id cannot be empty.")

        if not self.lineage_element_id.strip():
            raise ValueError("lineage_element_id cannot be empty.")

        if not self.chunk_ids:
            raise ValueError("At least one chunk ID is required.")

        if len(self.chunk_ids) != len(self.chunk_indexes):
            raise ValueError("Chunk IDs and indexes must have equal lengths.")


class GoldEvidenceMapper:
    """Resolve gold supporting facts through exact chunk source lineage."""

    def map(
        self,
        *,
        chunks: Iterable[DocumentChunk],
        supporting_facts: Iterable[SupportingFact],
        source_elements: Iterable[DocumentElement] = (),
    ) -> tuple[GoldEvidenceChunkMapping, ...]:
        """Map every supporting fact or fail when any source element is absent."""

        ordered_chunks = tuple(
            sorted(
                chunks,
                key=lambda chunk: (chunk.chunk_index, chunk.chunk_id),
            )
        )

        if not ordered_chunks:
            raise GoldEvidenceMappingError("At least one document chunk is required.")

        _validate_chunks(ordered_chunks)
        document_id = ordered_chunks[0].document_id
        ordered_facts = tuple(
            sorted(
                supporting_facts,
                key=lambda fact: (fact.source_key, fact.element_id),
            )
        )
        _validate_supporting_facts(ordered_facts)
        elements_by_id = _index_source_elements(
            source_elements,
            document_id=document_id,
        )

        chunks_by_element_id: dict[str, list[DocumentChunk]] = defaultdict(list)

        for chunk in ordered_chunks:
            for element_id in chunk.source_element_ids:
                chunks_by_element_id[element_id].append(chunk)

        mappings: list[GoldEvidenceChunkMapping] = []
        missing_element_ids: list[str] = []

        for fact in ordered_facts:
            matching_chunks = chunks_by_element_id.get(fact.element_id, [])
            lineage_element_id = fact.element_id

            if not matching_chunks:
                source_element = elements_by_id.get(fact.element_id)

                if source_element is not None and source_element.parent_element_id is not None:
                    lineage_element_id = source_element.parent_element_id
                    parent_chunks = chunks_by_element_id.get(
                        lineage_element_id,
                        [],
                    )
                    matching_chunks = _filter_chunks_by_source_text(
                        parent_chunks,
                        source_text=source_element.source_text,
                    )

            if not matching_chunks:
                missing_element_ids.append(fact.element_id)
                continue

            mappings.append(
                GoldEvidenceChunkMapping(
                    source_key=fact.source_key,
                    source_element_id=fact.element_id,
                    lineage_element_id=lineage_element_id,
                    chunk_ids=tuple(chunk.chunk_id for chunk in matching_chunks),
                    chunk_indexes=tuple(chunk.chunk_index for chunk in matching_chunks),
                )
            )

        if missing_element_ids:
            missing = ", ".join(repr(value) for value in missing_element_ids)
            raise GoldEvidenceMappingError(
                f"Gold evidence references source elements absent from chunks: {missing}."
            )

        return tuple(mappings)


def _validate_chunks(chunks: tuple[DocumentChunk, ...]) -> None:
    """Require one document and unique chunk identities."""

    document_ids = {chunk.document_id for chunk in chunks}

    if len(document_ids) != 1:
        raise GoldEvidenceMappingError("Chunks must belong to one document.")

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    if len(set(chunk_ids)) != len(chunk_ids):
        raise GoldEvidenceMappingError("Chunk IDs must be unique.")

    chunk_indexes = [chunk.chunk_index for chunk in chunks]

    if len(set(chunk_indexes)) != len(chunk_indexes):
        raise GoldEvidenceMappingError("Chunk indexes must be unique.")


def _validate_supporting_facts(
    supporting_facts: tuple[SupportingFact, ...],
) -> None:
    """Reject duplicate gold annotations before building mappings."""

    source_keys = [fact.source_key for fact in supporting_facts]

    if len(set(source_keys)) != len(source_keys):
        raise GoldEvidenceMappingError("Supporting facts must be unique.")


def _index_source_elements(
    source_elements: Iterable[DocumentElement],
    *,
    document_id: str,
) -> dict[str, DocumentElement]:
    """Index optional elements used for parent-scoped table-row matching."""

    indexed: dict[str, DocumentElement] = {}

    for element in source_elements:
        if element.document_id != document_id:
            raise GoldEvidenceMappingError(
                "Source elements and chunks must belong to one document."
            )

        if element.element_id in indexed:
            raise GoldEvidenceMappingError("Source element IDs must be unique.")

        indexed[element.element_id] = element

    return indexed


def _filter_chunks_by_source_text(
    chunks: Iterable[DocumentChunk],
    *,
    source_text: str,
) -> list[DocumentChunk]:
    """Find row text only inside chunks retaining its parent table lineage."""

    normalized_source_text = _normalize_evidence_text(source_text)

    if not normalized_source_text:
        return []

    return [
        chunk for chunk in chunks if normalized_source_text in _normalize_evidence_text(chunk.text)
    ]


def _normalize_evidence_text(value: str) -> str:
    """Normalize Markdown separators while preserving financial tokens."""

    normalized = _NON_EVIDENCE_TOKEN_PATTERN.sub(" ", value.casefold())
    return " ".join(normalized.split())
