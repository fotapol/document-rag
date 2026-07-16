from enum import StrEnum

from pydantic import Field, JsonValue

from document_rag.domain.base import BaseDomainModel


class QuestionType(StrEnum):
    """Supported financial question categories."""

    UNKNOWN = "unknown"
    LOOKUP = "lookup"
    COMPARISON = "comparison"
    CALCULATION = "calculation"
    TREND = "trend"


class Question(BaseDomainModel):
    """Question asked against a source document."""

    question_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    question_type: QuestionType = QuestionType.UNKNOWN
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
