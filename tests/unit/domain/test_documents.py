import pytest
from pydantic import ValidationError

from document_rag.domain import (
    Document,
    DocumentElement,
    DocumentElementType,
    TableCoordinates,
)


def test_document_serializes_to_json() -> None:
    document = Document(
        document_id="annual-report-2025",
        file_name="annual-report-2025.pdf",
        title="Annual Report 2025",
        page_count=120,
    )

    result = document.model_dump(mode="json")

    assert result["document_id"] == "annual-report-2025"
    assert result["page_count"] == 120


def test_document_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Document.model_validate(
            {
                "document_id": "report",
                "file_name": "report.pdf",
                "unexpected_field": True,
            }
        )


def test_table_row_requires_coordinates() -> None:
    with pytest.raises(ValidationError, match="require table coordinates"):
        DocumentElement(
            element_id="row-1",
            document_id="report",
            element_type=DocumentElementType.TABLE_ROW,
            source_text="Revenue | 2025 | 120",
            page_number=14,
        )


def test_table_row_accepts_coordinates() -> None:
    element = DocumentElement(
        element_id="row-1",
        document_id="report",
        element_type=DocumentElementType.TABLE_ROW,
        source_text="Revenue | 2025 | 120",
        page_number=14,
        table_coordinates=TableCoordinates(
            table_id="table-1",
            row_index=1,
        ),
    )

    assert element.table_coordinates is not None
    assert element.table_coordinates.row_index == 1


def test_paragraph_rejects_table_coordinates() -> None:
    with pytest.raises(ValidationError, match="Only table elements"):
        DocumentElement(
            element_id="paragraph-1",
            document_id="report",
            element_type=DocumentElementType.PARAGRAPH,
            source_text="Revenue increased during the reporting period.",
            page_number=14,
            table_coordinates=TableCoordinates(table_id="table-1"),
        )
