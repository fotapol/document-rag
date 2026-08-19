"""Tests for deterministic Markdown document chunking."""

import json

from document_rag.ingestion.chunking import (
    ChunkingConfig,
    MarkdownChunker,
    chunks_to_jsonl,
)
from document_rag.ingestion.llamaparse import (
    ParsedDocument,
    ParsedPage,
)


def build_document(markdown: str) -> ParsedDocument:
    """Create one deterministic parsed document fixture."""

    return ParsedDocument(
        document_id="sha256:test-document",
        filename="report.pdf",
        sha256="test-document",
        pages=(
            ParsedPage(
                page_number=1,
                markdown=markdown,
            ),
        ),
    )


def test_repeated_chunking_produces_identical_ids() -> None:
    """Identical input and configuration should produce identical chunks."""

    document = build_document(
        "# Revenue\n\nRevenue increased by 12%.\n\n# Expenses\n\nExpenses decreased by 3%."
    )
    chunker = MarkdownChunker(
        ChunkingConfig(
            target_chars=50,
            max_chars=100,
        )
    )

    first = chunker.chunk(document)
    second = chunker.chunk(document)

    assert first == second
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def test_chunks_preserve_document_and_page_metadata() -> None:
    """Every chunk should retain source document and page information."""

    document = build_document("# Summary\n\nNet income was $100.")
    chunks = MarkdownChunker().chunk(document)

    assert len(chunks) == 1
    assert chunks[0].document_id == document.document_id
    assert chunks[0].filename == document.filename
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1
    assert chunks[0].text
    assert chunks[0].char_count == len(chunks[0].text)


def test_small_markdown_table_remains_in_one_chunk() -> None:
    """A pipe table under the hard limit should remain intact."""

    table = "| Year | Revenue |\n| --- | ---: |\n| 2024 | 100 |\n| 2025 | 120 |"
    chunks = MarkdownChunker(
        ChunkingConfig(
            target_chars=500,
            max_chars=600,
        )
    ).chunk(build_document(table))

    assert len(chunks) == 1
    assert chunks[0].text == table


def test_large_markdown_table_repeats_header() -> None:
    """Each split pipe-table fragment should retain its header."""

    rows = "\n".join(f"| {year} | {'1' * 40} |" for year in range(2000, 2010))
    table = f"| Year | Revenue |\n| --- | ---: |\n{rows}"
    chunks = MarkdownChunker(
        ChunkingConfig(
            target_chars=120,
            max_chars=160,
        )
    ).chunk(build_document(table))

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.text.startswith("| Year | Revenue |\n| --- | ---: |")
        assert chunk.char_count <= 160


def test_large_markdown_table_repeats_heading_and_complete_rows() -> None:
    """A section heading must not make an oversized pipe table lose structure."""

    data_rows = [f"| {year} | {'1' * 40} |" for year in range(2000, 2010)]
    table = "# Revenue Table\n\n| Year | Revenue |\n| --- | ---: |\n" + "\n".join(data_rows)
    chunks = MarkdownChunker(
        ChunkingConfig(
            target_chars=120,
            max_chars=180,
        )
    ).chunk(build_document(table))

    assert len(chunks) > 1

    chunk_rows: list[str] = []

    for chunk in chunks:
        assert chunk.char_count <= 180
        assert chunk.text.startswith("# Revenue Table\n\n| Year | Revenue |\n| --- | ---: |")

        rows = [line for line in chunk.text.splitlines() if line.startswith("| 20")]
        assert rows
        assert all(row.endswith(" |") for row in rows)
        chunk_rows.extend(rows)

    assert chunk_rows == data_rows


def test_large_html_table_splits_only_between_rows() -> None:
    """LlamaParse HTML tables should retain context and table headers."""

    rows = "\n".join(
        (f"<tr><td>2025 Q{quarter}</td><td>Europe</td><td>${quarter * 100},000</td></tr>")
        for quarter in range(1, 9)
    )
    table = (
        "# Quarterly Revenue Detail\n\n"
        "<table>\n"
        "<thead><tr><th>Quarter</th><th>Region</th>"
        "<th>Revenue</th></tr></thead>\n"
        "<tbody>\n"
        f"{rows}\n"
        "</tbody>\n"
        "</table>"
    )
    chunks = MarkdownChunker(
        ChunkingConfig(
            target_chars=320,
            max_chars=420,
        )
    ).chunk(build_document(table))

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.char_count <= 420
        assert chunk.text.startswith("# Quarterly Revenue Detail\n\n<table>")
        assert "<thead>" in chunk.text
        assert "<th>Quarter</th>" in chunk.text
        assert chunk.text.endswith("</table>")
        assert chunk.text.count("<tr>") == chunk.text.count("</tr>")
        assert not chunk.text.lstrip().startswith("<td>")


def test_jsonl_export_contains_one_record_per_chunk() -> None:
    """JSONL export should preserve chunk order and identifiers."""

    document = build_document("# First\n\nOne.\n\n# Second\n\nTwo.")
    chunks = MarkdownChunker(
        ChunkingConfig(
            target_chars=20,
            max_chars=50,
        )
    ).chunk(document)

    lines = chunks_to_jsonl(chunks).splitlines()
    records = [json.loads(line) for line in lines]

    assert len(records) == len(chunks)
    assert [record["chunk_id"] for record in records] == [chunk.chunk_id for chunk in chunks]
    assert [record["chunk_index"] for record in records] == list(range(len(chunks)))
