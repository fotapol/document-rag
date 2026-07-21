"""Normalize linked DocFinQA records into deterministic records."""

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256

from document_rag.datasets.docfinqa.chunking import (
    DocFinQAChunker,
)
from document_rag.datasets.docfinqa.evidence import (
    DocFinQAEvidenceSelector,
)
from document_rag.datasets.docfinqa.linkage import (
    DocFinQALinkResult,
    DocFinQALinkStatus,
)
from document_rag.datasets.docfinqa.raw_models import (
    DocFinQARawRecord,
)
from document_rag.datasets.models import DatasetSplit

_ID_DIGEST_LENGTH = 32


class DocFinQANormalizationStatus(StrEnum):
    """Possible outcomes of normalizing one DocFinQA record."""

    NORMALIZED = "normalized"
    UNLINKED = "unlinked"
    EVIDENCE_INCOMPLETE = "evidence_incomplete"


@dataclass(frozen=True, slots=True)
class NormalizedDocFinQAElement:
    """One normalized retrieval element from a full report."""

    element_id: str
    document_id: str
    index: int
    start_char: int
    end_char: int
    source_text: str


@dataclass(frozen=True, slots=True)
class NormalizedDocFinQASupportingFact:
    """A gold FinQA fact linked to a DocFinQA element."""

    source_key: str
    element_id: str
    score: float


@dataclass(frozen=True, slots=True)
class NormalizedDocFinQARecord:
    """One fully normalized DocFinQA training example."""

    split: DatasetSplit
    document_id: str
    example_id: str
    finqa_id: str
    finqa_source_file: str
    link_status: DocFinQALinkStatus
    question: str
    answer: str
    program: str | None
    elements: tuple[NormalizedDocFinQAElement, ...]
    supporting_facts: tuple[
        NormalizedDocFinQASupportingFact,
        ...,
    ]


@dataclass(frozen=True, slots=True)
class DocFinQANormalizationResult:
    """Result returned for every input record, including skips."""

    status: DocFinQANormalizationStatus
    record: NormalizedDocFinQARecord | None
    expected_evidence_count: int
    matched_evidence_count: int


class DocFinQANormalizer:
    """Normalize one linked DocFinQA record.

    Records are rejected when linkage is unresolved or when not every
    FinQA gold supporting fact can be assigned to a DocFinQA chunk.

    IDs are derived from content rather than source order. They
    therefore remain stable when the input records are reordered.
    """

    def __init__(
        self,
        *,
        chunker: DocFinQAChunker | None = None,
        evidence_selector: (DocFinQAEvidenceSelector | None) = None,
    ) -> None:
        self._chunker = chunker or DocFinQAChunker()
        self._evidence_selector = evidence_selector or DocFinQAEvidenceSelector()

    def normalize(
        self,
        *,
        raw_record: DocFinQARawRecord,
        split: DatasetSplit,
        link_result: DocFinQALinkResult,
    ) -> DocFinQANormalizationResult:
        """Normalize a record or return a deterministic skip result."""

        finqa_record = link_result.finqa_record

        if finqa_record is None:
            return DocFinQANormalizationResult(
                status=(DocFinQANormalizationStatus.UNLINKED),
                record=None,
                expected_evidence_count=0,
                matched_evidence_count=0,
            )

        chunks = tuple(self._chunker.iter_chunks(raw_record.context))

        evidence_matches = self._evidence_selector.select(
            chunks=chunks,
            finqa_record=finqa_record,
        )

        expected_evidence_count = len(finqa_record.qa.gold_inds)
        matched_evidence_count = len(evidence_matches)

        if expected_evidence_count == 0 or matched_evidence_count != expected_evidence_count:
            return DocFinQANormalizationResult(
                status=(DocFinQANormalizationStatus.EVIDENCE_INCOMPLETE),
                record=None,
                expected_evidence_count=(expected_evidence_count),
                matched_evidence_count=(matched_evidence_count),
            )

        document_id = _build_document_id(raw_record.context)
        chunking_strategy = f"char-{self._chunker.chunk_size}-{self._chunker.overlap}"

        elements = tuple(
            NormalizedDocFinQAElement(
                element_id=_build_element_id(
                    document_id=document_id,
                    chunking_strategy=chunking_strategy,
                    chunk_index=chunk.index,
                ),
                document_id=document_id,
                index=chunk.index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                source_text=chunk.text,
            )
            for chunk in chunks
        )

        element_ids_by_index = {element.index: element.element_id for element in elements}

        supporting_facts = tuple(
            NormalizedDocFinQASupportingFact(
                source_key=match.source_key,
                element_id=element_ids_by_index[match.chunk_index],
                score=match.score,
            )
            for match in evidence_matches
        )

        question = raw_record.question.strip()
        answer = raw_record.answer.strip()
        program = raw_record.program.strip() or None

        example_id = _build_example_id(
            document_id=document_id,
            question=question,
            answer=answer,
            program=program,
        )

        normalized_record = NormalizedDocFinQARecord(
            split=split,
            document_id=document_id,
            example_id=example_id,
            finqa_id=finqa_record.id,
            finqa_source_file=finqa_record.filename,
            link_status=link_result.status,
            question=question,
            answer=answer,
            program=program,
            elements=elements,
            supporting_facts=supporting_facts,
        )

        return DocFinQANormalizationResult(
            status=(DocFinQANormalizationStatus.NORMALIZED),
            record=normalized_record,
            expected_evidence_count=(expected_evidence_count),
            matched_evidence_count=(matched_evidence_count),
        )


def _build_document_id(context: str) -> str:
    digest = _stable_digest(context)

    return f"docfinqa:document:{digest}"


def _build_example_id(
    *,
    document_id: str,
    question: str,
    answer: str,
    program: str | None,
) -> str:
    digest = _stable_digest(
        document_id,
        question,
        answer,
        program or "",
    )

    return f"docfinqa:example:{digest}"


def _build_element_id(
    *,
    document_id: str,
    chunking_strategy: str,
    chunk_index: int,
) -> str:
    return f"{document_id}:{chunking_strategy}:chunk-{chunk_index:06d}"


def _stable_digest(*values: str) -> str:
    """Hash length-prefixed UTF-8 values without ambiguity."""

    digest = sha256()

    for value in values:
        encoded_value = value.encode("utf-8")

        digest.update(
            len(encoded_value).to_bytes(
                length=8,
                byteorder="big",
            )
        )
        digest.update(encoded_value)

    return digest.hexdigest()[:_ID_DIGEST_LENGTH]
