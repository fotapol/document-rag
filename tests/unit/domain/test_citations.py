import pytest
from pydantic import ValidationError

from document_rag.domain import Citation


def test_citation_accepts_valid_source_reference() -> None:
    citation = Citation(
        citation_id="citation-1",
        document_id="report-2025",
        evidence_ids=("evidence-1", "evidence-2"),
        source_element_ids=("paragraph-10", "table-row-4"),
        page_numbers=(42, 43),
        excerpt="Revenue increased from $100 million to $120 million.",
        section="Results of Operations",
        table_id="table-3",
    )

    assert citation.page_numbers == (42, 43)
    assert citation.table_id == "table-3"


def test_citation_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        Citation(
            citation_id="citation-1",
            document_id="report-2025",
            evidence_ids=(),
            source_element_ids=("paragraph-10",),
            page_numbers=(42,),
            excerpt="Revenue increased.",
        )


@pytest.mark.parametrize(
    "page_numbers",
    [
        (42, 42),
        (43, 42),
    ],
)
def test_citation_rejects_invalid_page_order(
    page_numbers: tuple[int, ...],
) -> None:
    with pytest.raises(ValidationError, match="page numbers"):
        Citation(
            citation_id="citation-1",
            document_id="report-2025",
            evidence_ids=("evidence-1",),
            source_element_ids=("paragraph-10",),
            page_numbers=page_numbers,
            excerpt="Revenue increased.",
        )


def test_citation_serializes_tuples_as_json_arrays() -> None:
    citation = Citation(
        citation_id="citation-1",
        document_id="report-2025",
        evidence_ids=("evidence-1",),
        source_element_ids=("paragraph-10",),
        page_numbers=(42,),
        excerpt="Revenue increased.",
    )

    result = citation.model_dump(mode="json")

    assert result["evidence_ids"] == ["evidence-1"]
    assert result["page_numbers"] == [42]
