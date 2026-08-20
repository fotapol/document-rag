"""Tests for exact gold-evidence mapping through chunk lineage."""

import pytest

from document_rag.datasets.models import SupportingFact
from document_rag.domain.documents import (
    DocumentElement,
    DocumentElementType,
    TableCoordinates,
)
from document_rag.ingestion.chunking import (
    ChunkingConfig,
    DocumentChunk,
    MarkdownChunker,
)
from document_rag.ingestion.evidence import (
    GoldEvidenceMapper,
    GoldEvidenceMappingError,
)
from document_rag.ingestion.llamaparse import ParsedDocument, ParsedPage


def build_acceptance_chunks() -> tuple[DocumentChunk, ...]:
    """Chunk one lineaged financial section with an oversized table."""

    markdown = (
        "# Financial Results\n"
        "Revenue increased from 100 to 120 million dollars; "
        "Operating income | 40 | 55.\n\n"
        "| Metric | 2024 ($m) | 2025 ($m) |\n"
        "| --- | ---: | ---: |\n"
        "| Revenue | 100 | 120 |\n"
        "| Expenses | 60 | 65 |\n"
        "| Operating income | 40 | 55 |\n"
        "| Margin | 40% | 45.8% |"
    )
    document = ParsedDocument(
        document_id="report:2025",
        filename="report-2025.pdf",
        sha256="test-document",
        pages=(
            ParsedPage(
                page_number=7,
                markdown=markdown,
                source_element_ids=(
                    "heading:financial-results",
                    "paragraph:revenue-growth",
                    "table:financial-results",
                ),
            ),
        ),
    )

    return MarkdownChunker(
        ChunkingConfig(
            target_tokens=30,
            max_tokens=50,
            overlap_tokens=5,
        )
    ).chunk(document)


def build_acceptance_elements() -> tuple[DocumentElement, ...]:
    """Build the row-level gold element and its parent table."""

    table_id = "table:financial-results"
    return (
        DocumentElement(
            element_id=table_id,
            document_id="report:2025",
            element_type=DocumentElementType.TABLE,
            source_text="Financial results table",
            page_number=7,
            table_coordinates=TableCoordinates(table_id=table_id),
        ),
        DocumentElement(
            element_id="table-row:operating-income",
            document_id="report:2025",
            element_type=DocumentElementType.TABLE_ROW,
            source_text="Operating income | 40 | 55",
            page_number=7,
            parent_element_id=table_id,
            table_coordinates=TableCoordinates(
                table_id=table_id,
                row_index=3,
            ),
        ),
    )


def test_gold_evidence_maps_direct_and_row_elements_to_relevant_chunks() -> None:
    """Gold facts should resolve through exact or parent-scoped lineage."""

    chunks = build_acceptance_chunks()
    mappings = GoldEvidenceMapper().map(
        chunks=reversed(chunks),
        supporting_facts=(
            SupportingFact(
                source_key="text_0",
                element_id="paragraph:revenue-growth",
            ),
            SupportingFact(
                source_key="table_3",
                element_id="table-row:operating-income",
            ),
        ),
        source_elements=build_acceptance_elements(),
    )

    assert [mapping.source_key for mapping in mappings] == ["table_3", "text_0"]
    assert mappings[0].source_element_id == "table-row:operating-income"
    assert mappings[0].lineage_element_id == "table:financial-results"
    assert mappings[0].chunk_indexes == (2,)
    assert mappings[1].lineage_element_id == "paragraph:revenue-growth"
    assert mappings[1].chunk_indexes == (0,)
    assert all(chunk.page_start == chunk.page_end == 7 for chunk in chunks)
    assert all(chunk.token_count <= 50 for chunk in chunks)
    assert all(chunk.source_element_ids for chunk in chunks)

    chunks_by_index = {chunk.chunk_index: chunk for chunk in chunks}
    mapped_table_chunk = chunks_by_index[mappings[0].chunk_indexes[0]]

    assert "Operating income | 40 | 55" in chunks_by_index[0].text
    assert "| Operating income | 40 | 55 |" in mapped_table_chunk.text

    table_chunks = [
        chunk for chunk in chunks if "table:financial-results" in chunk.source_element_ids
    ]

    for chunk in table_chunks:
        assert chunk.text.startswith("# Financial Results")
        assert "| Metric | 2024 ($m) | 2025 ($m) |" in chunk.text

    data_rows = [
        line
        for chunk in table_chunks
        for line in chunk.text.splitlines()
        if line.startswith(
            (
                "| Revenue",
                "| Expenses",
                "| Operating income",
                "| Margin",
            )
        )
    ]
    assert data_rows == [
        "| Revenue | 100 | 120 |",
        "| Expenses | 60 | 65 |",
        "| Operating income | 40 | 55 |",
        "| Margin | 40% | 45.8% |",
    ]
    assert build_acceptance_chunks() == chunks


def test_gold_evidence_mapping_fails_for_missing_source_element() -> None:
    """Incomplete lineage must fail instead of silently dropping gold evidence."""

    with pytest.raises(
        GoldEvidenceMappingError,
        match=r"absent from chunks.*element:missing",
    ):
        GoldEvidenceMapper().map(
            chunks=build_acceptance_chunks(),
            supporting_facts=(
                SupportingFact(
                    source_key="text_missing",
                    element_id="element:missing",
                ),
            ),
        )


def test_gold_evidence_mapping_rejects_duplicate_facts() -> None:
    """Duplicate annotations should not create duplicate evaluation mappings."""

    fact = SupportingFact(
        source_key="text_0",
        element_id="paragraph:revenue-growth",
    )

    with pytest.raises(
        GoldEvidenceMappingError,
        match="Supporting facts must be unique",
    ):
        GoldEvidenceMapper().map(
            chunks=build_acceptance_chunks(),
            supporting_facts=(fact, fact),
        )


def test_gold_evidence_mapping_rejects_empty_chunks() -> None:
    """Mapping requires at least one chunk to be meaningful."""

    with pytest.raises(
        GoldEvidenceMappingError,
        match="At least one document chunk",
    ):
        GoldEvidenceMapper().map(
            chunks=(),
            supporting_facts=(),
        )
