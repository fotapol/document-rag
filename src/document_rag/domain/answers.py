from enum import StrEnum
from typing import Self

from pydantic import model_validator

from document_rag.domain.base import BaseDomainModel
from document_rag.domain.calculations import Calculation
from document_rag.domain.citations import Citation
from document_rag.domain.types import NonEmptyString


class AnswerStatus(StrEnum):
    """Outcome of answering a question."""

    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class Answer(BaseDomainModel):
    """Auditable answer produced by the RAG pipeline."""

    answer_id: NonEmptyString
    question_id: NonEmptyString
    text: NonEmptyString
    status: AnswerStatus

    citations: tuple[Citation, ...] = ()
    calculations: tuple[Calculation, ...] = ()

    @model_validator(mode="after")
    def validate_answer(self) -> Self:
        if self.status is AnswerStatus.ANSWERED and not self.citations:
            raise ValueError("Answered response requires at least one citation")

        citation_ids = [citation.citation_id for citation in self.citations]
        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("Citation IDs must be unique")

        calculation_ids = [calculation.calculation_id for calculation in self.calculations]
        if len(calculation_ids) != len(set(calculation_ids)):
            raise ValueError("Calculation IDs must be unique")

        if any(calculation.question_id != self.question_id for calculation in self.calculations):
            raise ValueError("All calculations must belong to the answer question")

        return self
