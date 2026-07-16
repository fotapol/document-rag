from typing import Annotated

from pydantic import Field, FiniteFloat, JsonValue

from document_rag.domain.base import BaseDomainModel

NonEmptyString = Annotated[str, Field(min_length=1)]
PositivePageNumber = Annotated[int, Field(ge=1)]


class Evidence(BaseDomainModel):
    """Exact context item selected as evidence for a question."""

    evidence_id: NonEmptyString
    question_id: NonEmptyString
    document_id: NonEmptyString
    text: NonEmptyString

    source_element_ids: tuple[NonEmptyString, ...] = Field(min_length=1)
    page_numbers: tuple[PositivePageNumber, ...] = Field(min_length=1)

    rank: int = Field(ge=1)
    retrieval_score: FiniteFloat | None = None
    reranker_score: FiniteFloat | None = None
    retriever: str | None = Field(default=None, min_length=1)

    metadata: dict[str, JsonValue] = Field(default_factory=dict)
