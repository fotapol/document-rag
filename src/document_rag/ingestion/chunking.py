"""Deterministic Markdown chunking for parsed documents."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256

from document_rag.ingestion.llamaparse import ParsedDocument

_TABLE_SEPARATOR_PATTERN = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*"
    r"(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")
_HTML_TABLE_START_PATTERN = re.compile(r"^\s*<table(?:\s|>)", re.IGNORECASE)
_HTML_TABLE_OPEN_PATTERN = re.compile(
    r"<table\b[^>]*>",
    re.IGNORECASE,
)
_HTML_TABLE_END_PATTERN = re.compile(
    r"</table>\s*$",
    re.IGNORECASE,
)
_HTML_THEAD_PATTERN = re.compile(
    r"<thead\b[^>]*>.*?</thead>",
    re.IGNORECASE | re.DOTALL,
)
_HTML_TBODY_PATTERN = re.compile(
    r"(?P<open><tbody\b[^>]*>)"
    r"(?P<body>.*?)"
    r"</tbody>",
    re.IGNORECASE | re.DOTALL,
)
_HTML_ROW_PATTERN = re.compile(
    r"<tr\b[^>]*>.*?</tr>",
    re.IGNORECASE | re.DOTALL,
)


class DocumentChunkingError(RuntimeError):
    """Raised when a parsed document cannot produce usable chunks."""


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Character-based limits for deterministic Markdown chunking."""

    target_chars: int = 1_200
    max_chars: int = 1_800

    def __post_init__(self) -> None:
        """Reject invalid limits before chunking begins."""

        if self.target_chars <= 0:
            raise ValueError("target_chars must be positive.")

        if self.max_chars < self.target_chars:
            raise ValueError("max_chars must be greater than or equal to target_chars.")


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """Normalized chunk with stable identity and source metadata."""

    chunk_id: str
    document_id: str
    document_sha256: str
    filename: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str
    char_count: int
    block_count: int

    def to_record(self) -> dict[str, object]:
        """Convert the chunk to a JSON-serializable record."""

        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_sha256": self.document_sha256,
            "filename": self.filename,
            "chunk_index": self.chunk_index,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "text": self.text,
            "char_count": self.char_count,
            "block_count": self.block_count,
        }


class MarkdownChunker:
    """Split page-level Markdown into stable retrieval chunks."""

    def __init__(
        self,
        config: ChunkingConfig | None = None,
    ) -> None:
        """Create a chunker with explicit or default limits."""

        self._config = config or ChunkingConfig()

    def chunk(
        self,
        document: ParsedDocument,
    ) -> tuple[DocumentChunk, ...]:
        """Chunk every page without crossing page boundaries."""

        chunks: list[DocumentChunk] = []
        chunk_index = 0

        for page in document.pages:
            normalized_markdown = _normalize_markdown(page.markdown)

            if not normalized_markdown:
                continue

            blocks = _split_markdown_blocks(normalized_markdown)
            blocks = _attach_headings(blocks)
            blocks = tuple(
                fragment
                for block in blocks
                for fragment in _split_oversized_block(
                    block,
                    self._config.max_chars,
                )
            )

            for chunk_blocks in _pack_blocks(
                blocks,
                target_chars=self._config.target_chars,
                max_chars=self._config.max_chars,
            ):
                text = "\n\n".join(chunk_blocks).strip()

                if not text:
                    continue

                chunk_id = _build_chunk_id(
                    document_id=document.document_id,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    text=text,
                )

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        document_sha256=document.sha256,
                        filename=document.filename,
                        chunk_index=chunk_index,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        text=text,
                        char_count=len(text),
                        block_count=len(chunk_blocks),
                    )
                )
                chunk_index += 1

        if not chunks:
            raise DocumentChunkingError("The parsed document produced no non-empty chunks.")

        return tuple(chunks)


def chunks_to_jsonl(chunks: Iterable[DocumentChunk]) -> str:
    """Serialize chunks as deterministic UTF-8 JSON Lines text."""

    records = [
        json.dumps(
            chunk.to_record(),
            ensure_ascii=False,
            sort_keys=True,
        )
        for chunk in chunks
    ]

    if not records:
        return ""

    return "\n".join(records) + "\n"


def _normalize_markdown(markdown: str) -> str:
    """Normalize line endings and trailing whitespace."""

    normalized = markdown.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.splitlines()]
    return "\n".join(lines).strip()


def _split_markdown_blocks(markdown: str) -> tuple[str, ...]:
    """Split Markdown into prose, fenced blocks, and table blocks."""

    lines = markdown.splitlines()
    blocks: list[str] = []
    index = 0

    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue

        if _is_fence_start(lines[index]):
            block, index = _consume_fenced_block(lines, index)
            blocks.append(block)
            continue

        if _is_html_table_start(lines[index]):
            block, index = _consume_html_table(lines, index)
            blocks.append(block)
            continue

        if _is_markdown_table_start(lines, index):
            block, index = _consume_markdown_table(lines, index)
            blocks.append(block)
            continue

        paragraph_lines: list[str] = []

        while index < len(lines):
            if not lines[index].strip():
                break

            if paragraph_lines and (
                _is_fence_start(lines[index])
                or _is_html_table_start(lines[index])
                or _is_markdown_table_start(lines, index)
            ):
                break

            paragraph_lines.append(lines[index])
            index += 1

        paragraph = "\n".join(paragraph_lines).strip()

        if paragraph:
            blocks.append(paragraph)

    return tuple(blocks)


def _attach_headings(blocks: tuple[str, ...]) -> tuple[str, ...]:
    """Attach a standalone Markdown heading to the following block."""

    attached: list[str] = []
    pending_heading: str | None = None

    for block in blocks:
        if _is_heading(block):
            if pending_heading is not None:
                attached.append(pending_heading)
            pending_heading = block
            continue

        if pending_heading is not None:
            attached.append(f"{pending_heading}\n\n{block}")
            pending_heading = None
        else:
            attached.append(block)

    if pending_heading is not None:
        attached.append(pending_heading)

    return tuple(attached)


def _split_oversized_block(
    block: str,
    max_chars: int,
) -> tuple[str, ...]:
    """Split an oversized block while preserving table structure."""

    if len(block) <= max_chars:
        return (block,)

    html_parts = _extract_html_table(block)

    if html_parts is not None:
        prefix, table = html_parts
        return _split_large_html_table(
            table,
            max_chars,
            prefix=prefix,
        )

    markdown_parts = _extract_markdown_table(block)

    if markdown_parts is not None:
        prefix, table = markdown_parts
        return _split_large_markdown_table(
            table,
            max_chars,
            prefix=prefix,
        )

    return _split_large_text(block, max_chars)


def _split_large_markdown_table(
    table: str,
    max_chars: int,
    *,
    prefix: str,
) -> tuple[str, ...]:
    """Split a large pipe table by rows and repeat its header."""

    lines = table.splitlines()

    if len(lines) < 3 or not _is_table_separator(lines[1]):
        return _split_large_text(table, max_chars)

    header = lines[:2]
    rows = lines[2:]
    fragments: list[str] = []
    current_rows: list[str] = []

    def render(fragment_rows: list[str]) -> str:
        """Render one complete table fragment with repeated context."""

        rendered_table = "\n".join([*header, *fragment_rows])

        if prefix:
            return f"{prefix}\n\n{rendered_table}"

        return rendered_table

    for row in rows:
        candidate = render([*current_rows, row])

        if current_rows and len(candidate) > max_chars:
            fragments.append(render(current_rows))
            current_rows = [row]
        else:
            current_rows.append(row)

        single_row_candidate = render(current_rows)

        if len(single_row_candidate) > max_chars:
            raise DocumentChunkingError("A Markdown table row exceeds max_chars.")

    if current_rows:
        fragments.append(render(current_rows))

    return tuple(fragment for fragment in fragments if fragment.strip())


def _split_large_html_table(
    table: str,
    max_chars: int,
    *,
    prefix: str,
) -> tuple[str, ...]:
    """Split an HTML table by complete rows and repeat its header."""

    table_open_match = _HTML_TABLE_OPEN_PATTERN.search(table)
    body_match = _HTML_TBODY_PATTERN.search(table)

    if table_open_match is None or body_match is None:
        raise DocumentChunkingError("An oversized HTML table has unsupported structure.")

    table_open = table_open_match.group(0).strip()
    thead_match = _HTML_THEAD_PATTERN.search(table)
    thead = thead_match.group(0).strip() if thead_match else ""
    tbody_open = body_match.group("open").strip()
    rows = [row.strip() for row in _HTML_ROW_PATTERN.findall(body_match.group("body"))]

    if not rows:
        raise DocumentChunkingError("An oversized HTML table contains no complete rows.")

    def render(fragment_rows: list[str]) -> str:
        """Render one complete table fragment with repeated context."""

        table_parts = [table_open]

        if thead:
            table_parts.append(thead)

        table_parts.extend(
            [
                tbody_open,
                "\n".join(fragment_rows),
                "</tbody>",
                "</table>",
            ]
        )
        rendered_table = "\n".join(table_parts)

        if prefix:
            return f"{prefix}\n\n{rendered_table}"

        return rendered_table

    fragments: list[str] = []
    current_rows: list[str] = []

    for row in rows:
        candidate = render([*current_rows, row])

        if current_rows and len(candidate) > max_chars:
            fragments.append(render(current_rows))
            current_rows = [row]
        else:
            current_rows.append(row)

        if len(render(current_rows)) > max_chars:
            raise DocumentChunkingError("An HTML table row exceeds max_chars.")

    if current_rows:
        fragments.append(render(current_rows))

    return tuple(fragments)


def _split_large_text(
    text: str,
    max_chars: int,
) -> tuple[str, ...]:
    """Split long prose by sentences and then by safe whitespace."""

    sentences = [
        sentence.strip() for sentence in _SENTENCE_BOUNDARY_PATTERN.split(text) if sentence.strip()
    ]

    if len(sentences) <= 1:
        return _hard_wrap(text, max_chars)

    fragments: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = sentence if not current else f"{current} {sentence}"

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            fragments.append(current)
            current = ""

        if len(sentence) <= max_chars:
            current = sentence
        else:
            fragments.extend(_hard_wrap(sentence, max_chars))

    if current:
        fragments.append(current)

    return tuple(fragments)


def _hard_wrap(
    text: str,
    max_chars: int,
) -> tuple[str, ...]:
    """Split text at nearby whitespace without losing characters."""

    remaining = text.strip()
    fragments: list[str] = []

    while len(remaining) > max_chars:
        split_at = remaining.rfind(" ", 0, max_chars + 1)

        if split_at <= 0:
            split_at = max_chars

        fragment = remaining[:split_at].strip()

        if fragment:
            fragments.append(fragment)

        remaining = remaining[split_at:].strip()

    if remaining:
        fragments.append(remaining)

    return tuple(fragments)


def _pack_blocks(
    blocks: tuple[str, ...],
    *,
    target_chars: int,
    max_chars: int,
) -> tuple[tuple[str, ...], ...]:
    """Pack blocks into chunks while enforcing the hard maximum."""

    packed: list[tuple[str, ...]] = []
    current: list[str] = []

    for block in blocks:
        candidate = "\n\n".join([*current, block])

        if current and len(candidate) > target_chars:
            packed.append(tuple(current))
            current = [block]
        else:
            current.append(block)

        current_text = "\n\n".join(current)

        if len(current_text) > max_chars:
            raise DocumentChunkingError("Internal chunking error: a chunk exceeded max_chars.")

    if current:
        packed.append(tuple(current))

    return tuple(packed)


def _build_chunk_id(
    *,
    document_id: str,
    page_number: int,
    chunk_index: int,
    text: str,
) -> str:
    """Build a stable chunk ID from source identity and normalized text."""

    payload = (f"{document_id}\0{page_number}\0{chunk_index}\0{text}").encode()
    digest = sha256(payload).hexdigest()
    return f"chunk:{digest[:24]}"


def _extract_html_table(
    block: str,
) -> tuple[str, str] | None:
    """Extract an HTML table and any preceding section context."""

    match = re.search(
        r"(?im)^\s*<table(?:\s|>)",
        block,
    )

    if match is None:
        return None

    prefix = block[: match.start()].strip()
    table = block[match.start() :].strip()

    if _HTML_TABLE_END_PATTERN.search(table) is None:
        return None

    return prefix, table


def _extract_markdown_table(
    block: str,
) -> tuple[str, str] | None:
    """Extract a pipe table and any preceding section context."""

    lines = block.splitlines()

    for index in range(len(lines) - 1):
        if "|" not in lines[index] or not _is_table_separator(lines[index + 1]):
            continue

        prefix = "\n".join(lines[:index]).strip()
        table = "\n".join(lines[index:]).strip()
        return prefix, table

    return None


def _is_heading(block: str) -> bool:
    """Return whether a block is one standalone ATX heading."""

    lines = block.splitlines()
    return len(lines) == 1 and bool(re.match(r"^\s{0,3}#{1,6}\s+\S", lines[0]))


def _is_fence_start(line: str) -> bool:
    """Return whether a line starts a fenced Markdown block."""

    stripped = line.lstrip()
    return stripped.startswith("```") or stripped.startswith("~~~")


def _consume_fenced_block(
    lines: list[str],
    start_index: int,
) -> tuple[str, int]:
    """Consume one fenced Markdown block."""

    opening = lines[start_index].lstrip()
    marker = opening[:3]
    collected = [lines[start_index]]
    index = start_index + 1

    while index < len(lines):
        collected.append(lines[index])

        if lines[index].lstrip().startswith(marker):
            index += 1
            break

        index += 1

    return "\n".join(collected).strip(), index


def _is_html_table_start(line: str) -> bool:
    """Return whether a line starts an HTML table."""

    return bool(_HTML_TABLE_START_PATTERN.match(line))


def _consume_html_table(
    lines: list[str],
    start_index: int,
) -> tuple[str, int]:
    """Consume one complete HTML table block."""

    collected: list[str] = []
    index = start_index

    while index < len(lines):
        collected.append(lines[index])
        index += 1

        if re.search(
            r"</table>\s*$",
            collected[-1],
            re.IGNORECASE,
        ):
            break

    return "\n".join(collected).strip(), index


def _is_markdown_table_start(
    lines: list[str],
    index: int,
) -> bool:
    """Return whether two lines begin a Markdown pipe table."""

    return index + 1 < len(lines) and "|" in lines[index] and _is_table_separator(lines[index + 1])


def _is_table_separator(line: str) -> bool:
    """Return whether a line is a Markdown table separator."""

    return bool(_TABLE_SEPARATOR_PATTERN.match(line))


def _consume_markdown_table(
    lines: list[str],
    start_index: int,
) -> tuple[str, int]:
    """Consume contiguous Markdown pipe-table rows."""

    collected = [
        lines[start_index],
        lines[start_index + 1],
    ]
    index = start_index + 2

    while index < len(lines):
        line = lines[index]

        if not line.strip() or "|" not in line:
            break

        collected.append(line)
        index += 1

    return "\n".join(collected).strip(), index
