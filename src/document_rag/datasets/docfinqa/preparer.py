"""Incremental preparation of normalized DocFinQA splits."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from document_rag.datasets.docfinqa.linkage import (
    DocFinQALinker,
    DocFinQALinkStatus,
)
from document_rag.datasets.docfinqa.normalizer import (
    DocFinQANormalizationStatus,
    DocFinQANormalizer,
)
from document_rag.datasets.docfinqa.raw_models import (
    DocFinQARawRecord,
)
from document_rag.datasets.finqa.raw_models import (
    FinQARawRecord,
)
from document_rag.datasets.models import (
    DatasetExample,
    DatasetSplit,
)
from document_rag.domain import Document, DocumentElement

PreparedDocFinQADocument = Document
PreparedDocFinQAExample = DatasetExample


@dataclass(frozen=True, slots=True)
class PreparedDocFinQAItem:
    """One item ready for incremental serialization.

    A document and its elements are included only on their first
    occurrence. Later questions referring to the same document emit
    only the example.
    """

    document: Document | None
    elements: tuple[DocumentElement, ...]
    example: DatasetExample


@dataclass(frozen=True, slots=True)
class DocFinQAPreparationStats:
    """Statistics collected while preparing one split."""

    total_records: int
    normalized_records: int
    unique_documents: int
    exact_links: int
    equivalent_links: int
    skipped_ambiguous: int
    skipped_answer_mismatch: int
    skipped_evidence_incomplete: int
    unmatched_evidence_facts: int
    skipped_duplicate: int

    @property
    def skipped_records(self) -> int:
        return (
            self.skipped_ambiguous
            + self.skipped_answer_mismatch
            + self.skipped_evidence_incomplete
            + self.skipped_duplicate
        )


class DocFinQASplitPreparer:
    """Prepare one DocFinQA split as a memory-bounded stream.

    The preparer links each DocFinQA record to FinQA, normalizes it,
    skips unresolved records and emits unique documents only once.

    The object is single-use because its document and example
    deduplication state belongs to one preparation run.
    """

    def __init__(
        self,
        *,
        split: DatasetSplit,
        finqa_records: Iterable[FinQARawRecord],
        normalizer: DocFinQANormalizer | None = None,
    ) -> None:
        self._split = split
        self._linker = DocFinQALinker(finqa_records)
        self._normalizer = normalizer or DocFinQANormalizer()

        self._seen_document_ids: set[str] = set()
        self._seen_example_ids: set[str] = set()
        self._started = False

        self._total_records = 0
        self._normalized_records = 0
        self._exact_links = 0
        self._equivalent_links = 0
        self._skipped_ambiguous = 0
        self._skipped_answer_mismatch = 0
        self._skipped_evidence_incomplete = 0
        self._unmatched_evidence_facts = 0
        self._skipped_duplicate = 0

    @property
    def stats(self) -> DocFinQAPreparationStats:
        """Return the current immutable statistics snapshot."""

        return DocFinQAPreparationStats(
            total_records=self._total_records,
            normalized_records=self._normalized_records,
            unique_documents=len(self._seen_document_ids),
            exact_links=self._exact_links,
            equivalent_links=self._equivalent_links,
            skipped_ambiguous=self._skipped_ambiguous,
            skipped_answer_mismatch=(self._skipped_answer_mismatch),
            skipped_evidence_incomplete=(self._skipped_evidence_incomplete),
            unmatched_evidence_facts=(self._unmatched_evidence_facts),
            skipped_duplicate=self._skipped_duplicate,
        )

    def iter_prepare(
        self,
        raw_records: Iterable[DocFinQARawRecord],
    ) -> Iterator[PreparedDocFinQAItem]:
        """Yield normalized records ready for incremental writing."""

        if self._started:
            raise RuntimeError("DocFinQASplitPreparer instances are single-use")

        self._started = True

        for raw_record in raw_records:
            self._total_records += 1

            link_result = self._linker.link(raw_record)

            if link_result.status is DocFinQALinkStatus.AMBIGUOUS:
                self._skipped_ambiguous += 1
                continue

            if link_result.status is DocFinQALinkStatus.ANSWER_MISMATCH:
                self._skipped_answer_mismatch += 1
                continue

            if link_result.status is DocFinQALinkStatus.EXACT:
                self._exact_links += 1
            elif link_result.status is DocFinQALinkStatus.EQUIVALENT:
                self._equivalent_links += 1
            else:
                raise RuntimeError(f"Unexpected DocFinQA link status: {link_result.status}")

            normalization_result = self._normalizer.normalize(
                raw_record=raw_record,
                split=self._split,
                link_result=link_result,
            )

            if normalization_result.status is (DocFinQANormalizationStatus.EVIDENCE_INCOMPLETE):
                self._skipped_evidence_incomplete += 1
                self._unmatched_evidence_facts += max(
                    0,
                    normalization_result.expected_evidence_count
                    - normalization_result.matched_evidence_count,
                )
                continue

            if (
                normalization_result.status is not DocFinQANormalizationStatus.NORMALIZED
                or normalization_result.record is None
            ):
                raise RuntimeError("Linked DocFinQA record was not normalized")

            normalized_record = normalization_result.record

            example_id = normalized_record.example.example_id

            if example_id in self._seen_example_ids:
                self._skipped_duplicate += 1
                continue

            self._seen_example_ids.add(example_id)

            document_id = normalized_record.document.document_id
            is_new_document = document_id not in self._seen_document_ids

            if is_new_document:
                self._seen_document_ids.add(document_id)
                document = normalized_record.document
                elements = normalized_record.elements
            else:
                document = None
                elements = ()

            self._normalized_records += 1

            yield PreparedDocFinQAItem(
                document=document,
                elements=elements,
                example=normalized_record.example,
            )

        if self._total_records == 0:
            raise ValueError(f"DocFinQA split is empty: {self._split.value}")
