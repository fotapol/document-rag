"""Tests for deterministic Markdown document chunking."""

import json

import pytest

from document_rag.ingestion.chunking import (
    ChunkingConfig,
    DocumentChunkingError,
    HuggingFaceTokenCounter,
    MarkdownChunker,
    RegexTokenCounter,
    chunks_to_jsonl,
)
from document_rag.ingestion.llamaparse import (
    ParsedDocument,
    ParsedPage,
)


class CharacterTokenCounter:
    """Treat characters as tokens to isolate structure-aware test limits."""

    def count(self, text: str) -> int:
        """Return the exact character length."""

        return len(text)


class ZeroTokenCounter:
    """Invalid counter used to verify defensive validation."""

    def count(self, text: str) -> int:
        """Incorrectly report zero for every input."""

        del text
        return 0


class FakeHuggingFaceTokenizer:
    """Minimal Hugging Face-compatible tokenizer test double."""

    def __init__(self) -> None:
        """Record how the adapter invokes tokenization."""

        self.add_special_tokens: bool | None = None

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> list[int]:
        """Return one synthetic ID per whitespace-delimited token."""

        self.add_special_tokens = add_special_tokens
        return list(range(len(text.split())))


def build_character_chunker(
    *,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int = 0,
) -> MarkdownChunker:
    """Build a chunker whose existing structure fixtures use exact lengths."""

    return MarkdownChunker(
        ChunkingConfig(
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        ),
        token_counter=CharacterTokenCounter(),
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
    chunker = build_character_chunker(
        target_tokens=50,
        max_tokens=100,
    )

    first = chunker.chunk(document)
    second = chunker.chunk(document)

    assert first == second
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def test_fixed_token_chunking_enforces_configured_limit() -> None:
    """The default counter should split prose at the configured token limit."""

    chunks = MarkdownChunker(
        ChunkingConfig(
            target_tokens=4,
            max_tokens=4,
            overlap_tokens=0,
        )
    ).chunk(build_document("Revenue grew rapidly. Expenses stayed flat."))

    assert [chunk.text for chunk in chunks] == [
        "Revenue grew rapidly.",
        "Expenses stayed flat.",
    ]
    assert [chunk.token_count for chunk in chunks] == [4, 4]


def test_hugging_face_counter_uses_model_tokens_without_special_tokens() -> None:
    """The adapter should use the supplied model tokenizer exactly."""

    tokenizer = FakeHuggingFaceTokenizer()
    counter = HuggingFaceTokenCounter(tokenizer)

    assert counter.count("revenue increased") == 2
    assert tokenizer.add_special_tokens is False


@pytest.mark.parametrize(
    ("target_tokens", "max_tokens", "message"),
    [
        (0, 1, "target_tokens must be positive"),
        (2, 1, "max_tokens must be greater than or equal to target_tokens"),
    ],
)
def test_chunking_config_rejects_invalid_token_limits(
    target_tokens: int,
    max_tokens: int,
    message: str,
) -> None:
    """Token limits should fail before document processing begins."""

    with pytest.raises(ValueError, match=message):
        ChunkingConfig(
            target_tokens=target_tokens,
            max_tokens=max_tokens,
        )


@pytest.mark.parametrize(
    ("overlap_tokens", "message"),
    [
        (-1, "overlap_tokens must be non-negative"),
        (4, "overlap_tokens must be smaller than target_tokens"),
    ],
)
def test_chunking_config_rejects_invalid_overlap(
    overlap_tokens: int,
    message: str,
) -> None:
    """Overlap must be bounded by the configured chunk target."""

    with pytest.raises(ValueError, match=message):
        ChunkingConfig(
            target_tokens=4,
            max_tokens=8,
            overlap_tokens=overlap_tokens,
        )


def test_chunker_rejects_counter_that_cannot_measure_text() -> None:
    """A broken injected counter must not silently bypass token limits."""

    chunker = MarkdownChunker(
        ChunkingConfig(
            target_tokens=1,
            max_tokens=1,
            overlap_tokens=0,
        ),
        token_counter=ZeroTokenCounter(),
    )

    with pytest.raises(
        DocumentChunkingError,
        match="token counter returned zero",
    ):
        chunker.chunk(build_document("Revenue"))


def test_chunks_preserve_document_and_page_metadata() -> None:
    """Every chunk should retain source document and page information."""

    document = build_document("# Summary\n\nNet income was $100.")
    chunks = MarkdownChunker().chunk(document)

    assert len(chunks) == 1
    assert chunks[0].document_id == document.document_id
    assert chunks[0].filename == document.filename
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1
    assert chunks[0].source_element_ids
    assert chunks[0].text
    assert chunks[0].char_count == len(chunks[0].text)
    assert chunks[0].token_count == RegexTokenCounter().count(chunks[0].text)


def test_small_markdown_table_remains_in_one_chunk() -> None:
    """A pipe table under the hard limit should remain intact."""

    table = "| Year | Revenue |\n| --- | ---: |\n| 2024 | 100 |\n| 2025 | 120 |"
    chunks = build_character_chunker(
        target_tokens=500,
        max_tokens=600,
    ).chunk(build_document(table))

    assert len(chunks) == 1
    assert chunks[0].text == table


def test_large_markdown_table_repeats_header() -> None:
    """Each split pipe-table fragment should retain its header."""

    rows = "\n".join(f"| {year} | {'1' * 40} |" for year in range(2000, 2010))
    table = f"| Year | Revenue |\n| --- | ---: |\n{rows}"
    chunks = build_character_chunker(
        target_tokens=120,
        max_tokens=160,
    ).chunk(build_document(table))

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.text.startswith("| Year | Revenue |\n| --- | ---: |")
        assert chunk.char_count <= 160


def test_large_markdown_table_repeats_heading_and_complete_rows() -> None:
    """A section heading must not make an oversized pipe table lose structure."""

    data_rows = [f"| {year} | {'1' * 40} |" for year in range(2000, 2010)]
    table = "# Revenue Table\n\n| Year | Revenue |\n| --- | ---: |\n" + "\n".join(data_rows)
    chunks = build_character_chunker(
        target_tokens=120,
        max_tokens=180,
        overlap_tokens=20,
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


def test_overlap_repeats_prose_within_the_same_section() -> None:
    """Later chunks should include bounded trailing prose and one heading."""

    document = build_document(
        "# Metrics\n\nalpha one two three.\n\nbeta four five six.\n\ngamma seven eight nine."
    )
    chunker = MarkdownChunker(
        ChunkingConfig(
            target_tokens=7,
            max_tokens=12,
            overlap_tokens=3,
        )
    )
    chunks = chunker.chunk(document)

    assert [chunk.text for chunk in chunks] == [
        "# Metrics\n\nalpha one two three.",
        "# Metrics\n\ntwo three.\n\nbeta four five six.",
        "# Metrics\n\nfive six.\n\ngamma seven eight nine.",
    ]
    assert all(chunk.text.count("# Metrics") == 1 for chunk in chunks)
    assert all(chunk.token_count <= 12 for chunk in chunks)
    assert len(chunks[1].source_element_ids) == 3
    assert chunker.chunk(document) == chunks


def test_overlap_does_not_cross_section_boundaries() -> None:
    """A new section must not inherit prose from the previous section."""

    chunks = MarkdownChunker(
        ChunkingConfig(
            target_tokens=7,
            max_tokens=12,
            overlap_tokens=3,
        )
    ).chunk(build_document("# Assets\nalpha one two three.\n\n# Liabilities\nbeta four five six."))

    assert len(chunks) == 2
    assert chunks[0].text.startswith("# Assets")
    assert chunks[1].text == "# Liabilities\n\nbeta four five six."
    assert "alpha" not in chunks[1].text


def test_nested_section_headings_are_propagated_to_every_chunk() -> None:
    """Parent and child headings should remain attached across a section."""

    chunks = MarkdownChunker(
        ChunkingConfig(
            target_tokens=8,
            max_tokens=12,
            overlap_tokens=0,
        )
    ).chunk(build_document("# Annual Report\n\n## Revenue\n\nalpha one two.\n\nbeta three four."))

    assert len(chunks) == 2
    assert all(chunk.text.startswith("# Annual Report\n\n## Revenue\n\n") for chunk in chunks)


def test_oversized_prose_fragments_repeat_their_section_heading() -> None:
    """Splitting one long paragraph must retain its heading on every fragment."""

    paragraph = " ".join(
        f"Revenue increased in reporting period {period}." for period in range(1, 6)
    )
    chunks = build_character_chunker(
        target_tokens=60,
        max_tokens=70,
    ).chunk(build_document(f"# Notes\n\n{paragraph}"))

    assert len(chunks) > 1
    assert all(chunk.text.startswith("# Notes\n\n") for chunk in chunks)
    assert all(chunk.char_count <= 70 for chunk in chunks)
    assert all(len(chunk.source_element_ids) == 2 for chunk in chunks)


def test_split_table_fragments_preserve_source_element_lineage() -> None:
    """Every derived table fragment should reference its original elements."""

    rows = "\n".join(f"| {year} | {'1' * 40} |" for year in range(2000, 2010))
    table = f"# Revenue Table\n\n| Year | Revenue |\n| --- | ---: |\n{rows}"
    chunker = build_character_chunker(
        target_tokens=120,
        max_tokens=180,
    )

    first = chunker.chunk(build_document(table))
    second = chunker.chunk(build_document(table))

    assert len(first) > 1
    assert all(len(chunk.source_element_ids) == 2 for chunk in first)
    assert all(chunk.source_element_ids == first[0].source_element_ids for chunk in first)
    assert [chunk.source_element_ids for chunk in first] == [
        chunk.source_element_ids for chunk in second
    ]


def test_source_element_ids_do_not_depend_on_chunk_limits() -> None:
    """Changing packing limits must not change source-element identity."""

    document = build_document(
        "# Summary\n\n"
        "Revenue increased substantially during the reporting period. "
        "Operating expenses remained stable."
    )
    wide_chunks = build_character_chunker(
        target_tokens=200,
        max_tokens=240,
    ).chunk(document)
    narrow_chunks = build_character_chunker(
        target_tokens=50,
        max_tokens=70,
    ).chunk(document)

    wide_element_ids = {
        element_id for chunk in wide_chunks for element_id in chunk.source_element_ids
    }
    narrow_element_ids = {
        element_id for chunk in narrow_chunks for element_id in chunk.source_element_ids
    }

    assert len(wide_chunks) == 1
    assert len(narrow_chunks) > 1
    assert narrow_element_ids == wide_element_ids


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
    chunks = build_character_chunker(
        target_tokens=320,
        max_tokens=420,
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
    chunks = build_character_chunker(
        target_tokens=20,
        max_tokens=50,
    ).chunk(document)

    lines = chunks_to_jsonl(chunks).splitlines()
    records = [json.loads(line) for line in lines]

    assert len(records) == len(chunks)
    assert [record["chunk_id"] for record in records] == [chunk.chunk_id for chunk in chunks]
    assert [record["chunk_index"] for record in records] == list(range(len(chunks)))
    assert [record["source_element_ids"] for record in records] == [
        list(chunk.source_element_ids) for chunk in chunks
    ]
    assert [record["token_count"] for record in records] == [chunk.token_count for chunk in chunks]
