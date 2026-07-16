from typing import Self

from pydantic import Field, model_validator

from document_rag.domain.base import BaseDomainModel
from document_rag.domain.types import NonEmptyString, PositivePageNumber


class Citation(BaseDomainModel):
    """Auditable source reference attached to a generated answer."""

    citation_id: NonEmptyString
    document_id: NonEmptyString

    evidence_ids: tuple[NonEmptyString, ...] = Field(min_length=1)
    source_element_ids: tuple[NonEmptyString, ...] = Field(min_length=1)
    page_numbers: tuple[PositivePageNumber, ...] = Field(min_length=1)

    excerpt: NonEmptyString
    section: str | None = Field(default=None, min_length=1)
    table_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_page_numbers(self) -> Self:
        if len(set(self.page_numbers)) != len(self.page_numbers):
            raise ValueError("Citation page numbers must be unique")

        if tuple(sorted(self.page_numbers)) != self.page_numbers:
            raise ValueError("Citation page numbers must be sorted")

        return self
