from enum import StrEnum

from pydantic import Field, FiniteFloat, JsonValue

from document_rag.domain.base import BaseDomainModel
from document_rag.domain.questions import Question
from document_rag.domain.types import NonEmptyString


class DatasetName(StrEnum):
    """Supported source datasets."""

    FINQA = "finqa"
    DOCFINQA = "docfinqa"


class DatasetSplit(StrEnum):
    """Normalized dataset splits."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class ReasoningStep(BaseDomainModel):
    """One annotated step from a reference calculation program."""

    operation: NonEmptyString
    arguments: tuple[NonEmptyString, ...] = Field(min_length=1)
    result: NonEmptyString | None = None


class ReferenceAnswer(BaseDomainModel):
    """Ground-truth answer supplied by a dataset."""

    text: NonEmptyString
    executable_answer: JsonValue | None = None
    explanation: str | None = Field(default=None, min_length=1)
    program: str | None = Field(default=None, min_length=1)
    normalized_program: str | None = Field(default=None, min_length=1)
    steps: tuple[ReasoningStep, ...] = ()


class SupportingFact(BaseDomainModel):
    """Reference from a question to a gold document element."""

    source_key: NonEmptyString
    element_id: NonEmptyString
    score: FiniteFloat | None = Field(default=None, ge=0.0, le=1.0)


class DatasetExample(BaseDomainModel):
    """Normalized question and its evaluation annotations."""

    dataset: DatasetName
    split: DatasetSplit
    example_id: NonEmptyString
    question: Question
    reference_answer: ReferenceAnswer
    supporting_facts: tuple[SupportingFact, ...] = Field(min_length=1)
