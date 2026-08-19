"""Deterministic Markdown chunking for parsed documents."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import pairwise
from typing import Protocol

from document_rag.ingestion.llamaparse import ParsedDocument

_TABLE_SEPARATOR_PATTERN = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*"
    r"(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
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


class TokenCounter(Protocol):
    """Count retrieval-model tokens without coupling to one tokenizer library."""

    def count(self, text: str) -> int:
        """Return the number of tokens in text."""


class HuggingFaceTokenizer(Protocol):
    """Minimal tokenizer API needed by the Hugging Face counter adapter."""

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> Sequence[int]:
        """Encode text into model token IDs."""


@dataclass(frozen=True, slots=True)
class HuggingFaceTokenCounter:
    """Count tokens with an injected Hugging Face-compatible tokenizer."""

    tokenizer: HuggingFaceTokenizer

    def count(self, text: str) -> int:
        """Count content tokens without model-level special tokens."""

        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )


@dataclass(frozen=True, slots=True)
class RegexTokenCounter:
    """Deterministic offline token counter for default ingestion."""

    def count(self, text: str) -> int:
        """Count Unicode word runs and individual punctuation symbols."""

        return len(_TOKEN_PATTERN.findall(text))


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Token-based limits for deterministic Markdown chunking."""

    target_tokens: int = 300
    max_tokens: int = 450
    overlap_tokens: int = 50

    def __post_init__(self) -> None:
        """Reject invalid limits before chunking begins."""

        if self.target_tokens <= 0:
            raise ValueError("target_tokens must be positive.")

        if self.max_tokens < self.target_tokens:
            raise ValueError("max_tokens must be greater than or equal to target_tokens.")

        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative.")

        if self.overlap_tokens >= self.target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens.")


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
    source_element_ids: tuple[str, ...]
    text: str
    char_count: int
    token_count: int
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
            "source_element_ids": list(self.source_element_ids),
            "text": self.text,
            "char_count": self.char_count,
            "token_count": self.token_count,
            "block_count": self.block_count,
        }


@dataclass(frozen=True, slots=True)
class _LineagedBlock:
    """Markdown block paired with the source elements that produced it."""

    text: str
    source_element_ids: tuple[str, ...]
    section_text: str = ""
    section_element_ids: tuple[str, ...] = ()
    overlap_eligible: bool = True


class MarkdownChunker:
    """Split page-level Markdown into stable retrieval chunks."""

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        *,
        token_counter: TokenCounter | None = None,
    ) -> None:
        """Create a chunker with explicit or default limits."""

        self._config = config or ChunkingConfig()
        self._token_counter = token_counter or RegexTokenCounter()

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

            blocks = _build_lineaged_blocks(
                _split_markdown_blocks(normalized_markdown),
                document_id=document.document_id,
                page_number=page.page_number,
            )
            blocks = _attach_headings(blocks)
            blocks = tuple(
                fragment
                for block in blocks
                for fragment in _split_oversized_block(
                    block,
                    self._config.max_tokens,
                    token_counter=self._token_counter,
                )
            )

            for chunk_blocks in _pack_blocks(
                blocks,
                target_tokens=self._config.target_tokens,
                max_tokens=self._config.max_tokens,
                overlap_tokens=self._config.overlap_tokens,
                token_counter=self._token_counter,
            ):
                text = _render_blocks(chunk_blocks)

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
                        source_element_ids=_collect_source_element_ids(chunk_blocks),
                        text=text,
                        char_count=len(text),
                        token_count=_count_tokens(text, self._token_counter),
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


def _join_context(context: str, text: str) -> str:
    """Join optional section context and block text exactly once."""

    if context:
        return f"{context}\n\n{text}".strip()

    return text.strip()


def _render_blocks(blocks: tuple[_LineagedBlock, ...]) -> str:
    """Render blocks while emitting each active section heading once."""

    rendered: list[str] = []
    active_section_ids: tuple[str, ...] | None = None

    for block in blocks:
        if block.section_element_ids != active_section_ids:
            if block.section_text:
                rendered.append(block.section_text)

            active_section_ids = block.section_element_ids

        rendered.append(block.text)

    return "\n\n".join(rendered).strip()


def _build_lineaged_blocks(
    blocks: tuple[str, ...],
    *,
    document_id: str,
    page_number: int,
) -> tuple[_LineagedBlock, ...]:
    """Assign stable source-element IDs before any chunk transformations."""

    return tuple(
        _LineagedBlock(
            text=block,
            source_element_ids=(
                _build_source_element_id(
                    document_id=document_id,
                    page_number=page_number,
                    element_index=element_index,
                    text=block,
                ),
            ),
            overlap_eligible=_is_overlap_eligible(block),
        )
        for element_index, block in enumerate(blocks)
    )


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

        if _is_heading(lines[index]):
            blocks.append(lines[index].strip())
            index += 1
            continue

        paragraph_lines: list[str] = []

        while index < len(lines):
            if not lines[index].strip():
                break

            if paragraph_lines and (
                _is_fence_start(lines[index])
                or _is_html_table_start(lines[index])
                or _is_markdown_table_start(lines, index)
                or _is_heading(lines[index])
            ):
                break

            paragraph_lines.append(lines[index])
            index += 1

        paragraph = "\n".join(paragraph_lines).strip()

        if paragraph:
            blocks.append(paragraph)

    return tuple(blocks)


def _attach_headings(
    blocks: tuple[_LineagedBlock, ...],
) -> tuple[_LineagedBlock, ...]:
    """Attach active Markdown heading context to every section block."""

    attached: list[_LineagedBlock] = []
    active_headings: list[tuple[int, _LineagedBlock]] = []
    headings_need_content = False

    for block in blocks:
        heading_level = _heading_level(block.text)

        if heading_level is not None:
            active_headings = [heading for heading in active_headings if heading[0] < heading_level]
            active_headings.append((heading_level, block))
            headings_need_content = True
            continue

        section_blocks = tuple(heading[1] for heading in active_headings)
        attached.append(
            replace(
                block,
                section_text="\n\n".join(heading.text for heading in section_blocks),
                section_element_ids=_collect_source_element_ids(section_blocks),
            )
        )
        headings_need_content = False

    if headings_need_content:
        section_blocks = tuple(heading[1] for heading in active_headings)
        attached.append(
            _LineagedBlock(
                text="\n\n".join(heading.text for heading in section_blocks),
                source_element_ids=_collect_source_element_ids(section_blocks),
                overlap_eligible=False,
            )
        )

    return tuple(attached)


def _split_oversized_block(
    block: _LineagedBlock,
    max_tokens: int,
    *,
    token_counter: TokenCounter,
) -> tuple[_LineagedBlock, ...]:
    """Split an oversized block while preserving table structure."""

    if _count_tokens(_render_blocks((block,)), token_counter) <= max_tokens:
        return (block,)

    html_parts = _extract_html_table(block.text)

    if html_parts is not None:
        prefix, table = html_parts
        fragments = _split_large_html_table(
            table,
            max_tokens,
            prefix=_join_context(block.section_text, prefix),
            token_counter=token_counter,
        )
    else:
        markdown_parts = _extract_markdown_table(block.text)

        if markdown_parts is not None:
            prefix, table = markdown_parts
            fragments = _split_large_markdown_table(
                table,
                max_tokens,
                prefix=_join_context(block.section_text, prefix),
                token_counter=token_counter,
            )
        else:
            fragments = _split_large_text(
                block.text,
                max_tokens,
                prefix=block.section_text,
                token_counter=token_counter,
            )

    return tuple(replace(block, text=fragment) for fragment in fragments)


def _split_large_markdown_table(
    table: str,
    max_tokens: int,
    *,
    prefix: str,
    token_counter: TokenCounter,
) -> tuple[str, ...]:
    """Split a large pipe table by rows and repeat its header."""

    lines = table.splitlines()

    if len(lines) < 3 or not _is_table_separator(lines[1]):
        return _split_large_text(
            table,
            max_tokens,
            prefix=prefix,
            token_counter=token_counter,
        )

    header = lines[:2]
    rows = lines[2:]
    fragments: list[str] = []
    current_rows: list[str] = []

    def render_table(fragment_rows: list[str]) -> str:
        """Render one complete table fragment."""

        return "\n".join([*header, *fragment_rows])

    def render(fragment_rows: list[str]) -> str:
        """Render one complete table fragment with section context."""

        return _join_context(prefix, render_table(fragment_rows))

    for row in rows:
        candidate = render([*current_rows, row])

        if current_rows and _count_tokens(candidate, token_counter) > max_tokens:
            fragments.append(render_table(current_rows))
            current_rows = [row]
        else:
            current_rows.append(row)

        single_row_candidate = render(current_rows)

        if _count_tokens(single_row_candidate, token_counter) > max_tokens:
            raise DocumentChunkingError("A Markdown table row exceeds max_tokens.")

    if current_rows:
        fragments.append(render_table(current_rows))

    return tuple(fragment for fragment in fragments if fragment.strip())


def _split_large_html_table(
    table: str,
    max_tokens: int,
    *,
    prefix: str,
    token_counter: TokenCounter,
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

    def render_table(fragment_rows: list[str]) -> str:
        """Render one complete table fragment."""

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
        return "\n".join(table_parts)

    def render(fragment_rows: list[str]) -> str:
        """Render one complete table fragment with section context."""

        return _join_context(prefix, render_table(fragment_rows))

    fragments: list[str] = []
    current_rows: list[str] = []

    for row in rows:
        candidate = render([*current_rows, row])

        if current_rows and _count_tokens(candidate, token_counter) > max_tokens:
            fragments.append(render_table(current_rows))
            current_rows = [row]
        else:
            current_rows.append(row)

        if _count_tokens(render(current_rows), token_counter) > max_tokens:
            raise DocumentChunkingError("An HTML table row exceeds max_tokens.")

    if current_rows:
        fragments.append(render_table(current_rows))

    return tuple(fragments)


def _split_large_text(
    text: str,
    max_tokens: int,
    *,
    prefix: str,
    token_counter: TokenCounter,
) -> tuple[str, ...]:
    """Split long prose while reserving room for repeated section context."""

    sentences = [
        sentence.strip() for sentence in _SENTENCE_BOUNDARY_PATTERN.split(text) if sentence.strip()
    ]

    if len(sentences) <= 1:
        return _hard_wrap(
            text,
            max_tokens,
            prefix=prefix,
            token_counter=token_counter,
        )

    fragments: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = sentence if not current else f"{current} {sentence}"

        rendered_candidate = _join_context(prefix, candidate)

        if _count_tokens(rendered_candidate, token_counter) <= max_tokens:
            current = candidate
            continue

        if current:
            fragments.append(current)
            current = ""

        rendered_sentence = _join_context(prefix, sentence)

        if _count_tokens(rendered_sentence, token_counter) <= max_tokens:
            current = sentence
        else:
            fragments.extend(
                _hard_wrap(
                    sentence,
                    max_tokens,
                    prefix=prefix,
                    token_counter=token_counter,
                )
            )

    if current:
        fragments.append(current)

    return tuple(fragments)


def _hard_wrap(
    text: str,
    max_tokens: int,
    *,
    prefix: str,
    token_counter: TokenCounter,
) -> tuple[str, ...]:
    """Split text at nearby whitespace while enforcing a token maximum."""

    remaining = text.strip()
    fragments: list[str] = []

    while (
        _count_tokens(
            _join_context(prefix, remaining),
            token_counter,
        )
        > max_tokens
    ):
        split_at = _largest_prefix_within_token_limit(
            remaining,
            max_tokens,
            prefix=prefix,
            token_counter=token_counter,
        )
        whitespace_split = _last_whitespace_before(remaining, split_at)

        if whitespace_split > 0:
            split_at = whitespace_split

        fragment = remaining[:split_at].strip()

        if fragment:
            fragments.append(fragment)

        remaining = remaining[split_at:].strip()

    if remaining:
        fragments.append(remaining)

    return tuple(fragments)


def _largest_prefix_within_token_limit(
    text: str,
    max_tokens: int,
    *,
    prefix: str,
    token_counter: TokenCounter,
) -> int:
    """Find the longest character prefix accepted by the token counter."""

    lower = 1
    upper = len(text)
    best = 0

    while lower <= upper:
        midpoint = (lower + upper) // 2
        candidate = text[:midpoint].rstrip()
        rendered_candidate = _join_context(prefix, candidate)

        if candidate and _count_tokens(rendered_candidate, token_counter) <= max_tokens:
            best = midpoint
            lower = midpoint + 1
        else:
            upper = midpoint - 1

    if best == 0:
        raise DocumentChunkingError("The token counter cannot fit any text within max_tokens.")

    return best


def _last_whitespace_before(text: str, end: int) -> int:
    """Return the last whitespace position before a candidate split."""

    for index in range(min(end, len(text) - 1), 0, -1):
        if text[index].isspace():
            return index

    return -1


def _count_tokens(
    text: str,
    token_counter: TokenCounter,
) -> int:
    """Count tokens and reject counters that cannot measure non-empty text."""

    token_count = token_counter.count(text)

    if token_count < 0:
        raise DocumentChunkingError("The token counter returned a negative count.")

    if text.strip() and token_count == 0:
        raise DocumentChunkingError("The token counter returned zero for non-empty text.")

    return token_count


def _pack_blocks(
    blocks: tuple[_LineagedBlock, ...],
    *,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
    token_counter: TokenCounter,
) -> tuple[tuple[_LineagedBlock, ...], ...]:
    """Pack blocks and add same-section prose overlap between chunks."""

    packed: list[tuple[_LineagedBlock, ...]] = []
    current: list[_LineagedBlock] = []

    for block in blocks:
        candidate_blocks = (*current, block)
        candidate = _render_blocks(candidate_blocks)

        if current and _count_tokens(candidate, token_counter) > target_tokens:
            packed.append(tuple(current))
            current = [block]
        else:
            current.append(block)

        current_text = _render_blocks(tuple(current))

        if _count_tokens(current_text, token_counter) > max_tokens:
            raise DocumentChunkingError("Internal chunking error: a chunk exceeded max_tokens.")

    if current:
        packed.append(tuple(current))

    return _apply_overlap(
        tuple(packed),
        overlap_tokens=overlap_tokens,
        max_tokens=max_tokens,
        token_counter=token_counter,
    )


def _apply_overlap(
    packed: tuple[tuple[_LineagedBlock, ...], ...],
    *,
    overlap_tokens: int,
    max_tokens: int,
    token_counter: TokenCounter,
) -> tuple[tuple[_LineagedBlock, ...], ...]:
    """Add bounded overlap without crossing sections or structural blocks."""

    if overlap_tokens == 0 or len(packed) < 2:
        return packed

    overlapped = [packed[0]]

    for previous, current in pairwise(packed):
        overlap = _select_overlap_blocks(
            previous,
            current,
            overlap_tokens=overlap_tokens,
            max_tokens=max_tokens,
            token_counter=token_counter,
        )
        overlapped.append((*overlap, *current))

    return tuple(overlapped)


def _select_overlap_blocks(
    previous: tuple[_LineagedBlock, ...],
    current: tuple[_LineagedBlock, ...],
    *,
    overlap_tokens: int,
    max_tokens: int,
    token_counter: TokenCounter,
) -> tuple[_LineagedBlock, ...]:
    """Select the largest eligible suffix that fits the overlap budget."""

    first_current = current[0]

    if not first_current.overlap_eligible:
        return ()

    section_element_ids = first_current.section_element_ids
    selected: list[_LineagedBlock] = []

    for block in reversed(previous):
        if not block.overlap_eligible or block.section_element_ids != section_element_ids:
            break

        candidate = (block, *selected)

        if _overlap_fits(
            candidate,
            current,
            overlap_tokens=overlap_tokens,
            max_tokens=max_tokens,
            token_counter=token_counter,
        ):
            selected.insert(0, block)
            continue

        trimmed = _trim_block_for_overlap(
            block,
            tuple(selected),
            current,
            overlap_tokens=overlap_tokens,
            max_tokens=max_tokens,
            token_counter=token_counter,
        )

        if trimmed is not None:
            selected.insert(0, trimmed)

        break

    return tuple(selected)


def _trim_block_for_overlap(
    block: _LineagedBlock,
    selected: tuple[_LineagedBlock, ...],
    current: tuple[_LineagedBlock, ...],
    *,
    overlap_tokens: int,
    max_tokens: int,
    token_counter: TokenCounter,
) -> _LineagedBlock | None:
    """Trim one prose block to its longest suffix that fits overlap."""

    text = block.text.strip()
    whitespace_starts = [
        match.end() for match in re.finditer(r"\s+", text) if match.end() < len(text)
    ]
    candidate_starts = [*whitespace_starts, *range(1, len(text))]
    seen_starts: set[int] = set()

    for start in candidate_starts:
        if start in seen_starts:
            continue

        seen_starts.add(start)
        suffix = text[start:].strip()

        if not suffix:
            continue

        trimmed = replace(block, text=suffix)
        overlap = (trimmed, *selected)

        if _overlap_fits(
            overlap,
            current,
            overlap_tokens=overlap_tokens,
            max_tokens=max_tokens,
            token_counter=token_counter,
        ):
            return trimmed

    return None


def _overlap_fits(
    overlap: tuple[_LineagedBlock, ...],
    current: tuple[_LineagedBlock, ...],
    *,
    overlap_tokens: int,
    max_tokens: int,
    token_counter: TokenCounter,
) -> bool:
    """Return whether overlap fits both its budget and the chunk maximum."""

    current_token_count = _count_tokens(
        _render_blocks(current),
        token_counter,
    )
    combined_token_count = _count_tokens(
        _render_blocks((*overlap, *current)),
        token_counter,
    )
    added_token_count = max(0, combined_token_count - current_token_count)

    return added_token_count <= overlap_tokens and combined_token_count <= max_tokens


def _collect_source_element_ids(
    blocks: tuple[_LineagedBlock, ...],
) -> tuple[str, ...]:
    """Collect source-element IDs in first-occurrence order."""

    return tuple(
        dict.fromkeys(
            element_id
            for block in blocks
            for element_id in (
                *block.section_element_ids,
                *block.source_element_ids,
            )
        )
    )


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


def _build_source_element_id(
    *,
    document_id: str,
    page_number: int,
    element_index: int,
    text: str,
) -> str:
    """Build a stable ID for one normalized source Markdown block."""

    payload = (f"{document_id}\0{page_number}\0{element_index}\0{text}").encode()
    digest = sha256(payload).hexdigest()
    return f"element:{digest[:24]}"


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

    return _heading_level(block) is not None


def _heading_level(block: str) -> int | None:
    """Return the ATX heading level for one standalone heading block."""

    lines = block.splitlines()

    if len(lines) != 1:
        return None

    match = re.match(r"^\s{0,3}(#{1,6})\s+\S", lines[0])
    return len(match.group(1)) if match is not None else None


def _is_overlap_eligible(block: str) -> bool:
    """Return whether a source block may be repeated as prose overlap."""

    first_line = block.splitlines()[0]
    return (
        not _is_heading(block)
        and not _is_fence_start(first_line)
        and _extract_html_table(block) is None
        and _extract_markdown_table(block) is None
    )


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
