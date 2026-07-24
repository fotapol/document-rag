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
)
from document_rag.datasets.docfinqa.raw_models import (
    DocFinQARawRecord,
)
from document_rag.datasets.models import (
    DatasetExample,
    DatasetName,
    DatasetSplit,
    ReferenceAnswer,
    SupportingFact,
)
from document_rag.domain import (
    Document,
    DocumentElement,
    DocumentElementType,
    Question,
)

_ID_DIGEST_LENGTH = 32


class DocFinQANormalizationStatus(StrEnum):
    """Possible outcomes of normalizing one DocFinQA record."""

    NORMALIZED = "normalized"
    UNLINKED = "unlinked"
    EVIDENCE_INCOMPLETE = "evidence_incomplete"


NormalizedDocFinQAElement = DocumentElement
NormalizedDocFinQASupportingFact = SupportingFact


@dataclass(frozen=True, slots=True)
class NormalizedDocFinQARecord:
    """One DocFinQA record normalized into shared domain models."""

    document: Document
    elements: tuple[DocumentElement, ...]
    example: DatasetExample


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

        document = Document(
            document_id=document_id,
            file_name=f"{document_id.rsplit(':', maxsplit=1)[-1]}.txt",
            mime_type="text/plain",
            checksum_sha256=sha256(raw_record.context.encode("utf-8")).hexdigest(),
            page_count=1,
            metadata={
                "dataset": DatasetName.DOCFINQA.value,
                "split": split.value,
                "pagination": "unavailable",
            },
        )

        elements = tuple(
            DocumentElement(
                element_id=_build_element_id(
                    document_id=document_id,
                    chunking_strategy=chunking_strategy,
                    chunk_index=chunk.index,
                ),
                document_id=document_id,
                element_type=DocumentElementType.PARAGRAPH,
                source_text=chunk.text,
                page_number=1,
                metadata={
                    "dataset": DatasetName.DOCFINQA.value,
                    "split": split.value,
                    "chunk_index": chunk.index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "chunking_strategy": chunking_strategy,
                },
            )
            for chunk in chunks
        )

        element_ids_by_index = {
            chunk.index: element.element_id
            for chunk, element in zip(
                chunks,
                elements,
                strict=True,
            )
        }

        supporting_facts = tuple(
            SupportingFact(
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

        normalized_question = Question(
            question_id=example_id,
            document_id=document_id,
            text=question,
            metadata={
                "dataset": DatasetName.DOCFINQA.value,
                "source_example_id": finqa_record.id,
                "source_file": finqa_record.filename,
                "link_status": link_result.status.value,
            },
        )

        example = DatasetExample(
            dataset=DatasetName.DOCFINQA,
            split=split,
            example_id=example_id,
            question=normalized_question,
            reference_answer=ReferenceAnswer(
                text=answer,
                program=program,
            ),
            supporting_facts=supporting_facts,
        )

        normalized_record = NormalizedDocFinQARecord(
            document=document,
            elements=elements,
            example=example,
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
