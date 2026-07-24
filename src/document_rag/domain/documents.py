from enum import StrEnum
from typing import Annotated, Self

from pydantic import Field, JsonValue, StringConstraints, model_validator

from document_rag.domain.base import BaseDomainModel


class DocumentElementType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    TABLE_ROW = "table_row"
    CAPTION = "caption"
    FOOTNOTE = "footnote"


class Document(BaseDomainModel):
    """Source document processed by the RAG pipeline."""

    document_id: str = Field(min_length=1)
    file_name: str = Field(min_length=1)
    title: str | None = Field(default=None, min_length=1)
    mime_type: str = Field(default="application/pdf", min_length=1)
    source_uri: str | None = Field(default=None, min_length=1)
    checksum_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    page_count: int | None = Field(default=None, ge=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class TableCoordinates(BaseDomainModel):
    table_id: str = Field(min_length=1)
    row_index: int | None = Field(default=None, ge=0)
    column_index: int | None = Field(default=None, ge=0)
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)


class DocumentElement(BaseDomainModel):
    element_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    element_type: DocumentElementType
    source_text: Annotated[
        str,
        StringConstraints(
            min_length=1,
            strip_whitespace=False,
        ),
    ]
    page_number: int = Field(ge=1)
    section: str | None = Field(default=None, min_length=1)
    parent_element_id: str | None = Field(default=None, min_length=1)
    table_coordinates: TableCoordinates | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_table_coordinates(self) -> Self:
        table_types = {
            DocumentElementType.TABLE,
            DocumentElementType.TABLE_ROW,
        }

        if self.element_type in table_types and self.table_coordinates is None:
            raise ValueError("Table elements require table coordinates")

        if self.element_type not in table_types and self.table_coordinates is not None:
            raise ValueError("Only table elements may have table coordinates")

        return self
