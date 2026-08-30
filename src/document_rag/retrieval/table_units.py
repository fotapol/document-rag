"""Canonical table-row retrieval units with recoverable parent lineage."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from html.parser import HTMLParser
from typing import Protocol

from document_rag.ingestion.chunking import DocumentChunk, RegexTokenCounter
from document_rag.retrieval.models import RetrievalResult

_HTML_TABLE_PATTERN = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
_MARKDOWN_SEPARATOR_PATTERN = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
_MARKDOWN_HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+(?P<title>.+?)\s*#*\s*$")
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_NON_CONTENT_PATTERN = re.compile(r"[^\w]+", re.UNICODE)


class RankedRetriever(Protocol):
    """Search boundary needed by the parent-aware wrapper."""

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> tuple[RetrievalResult, ...]:
        """Return a deterministic ranking for one query."""


@dataclass(frozen=True, slots=True)
class _ParsedTable:
    """One normalized table before it is expanded into row chunks."""

    start: int
    end: int
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


class TableRetrievalCorpus:
    """Retrieval units plus the original chunks needed for parent recovery."""

    def __init__(
        self,
        *,
        units: Iterable[DocumentChunk],
        parents: Iterable[DocumentChunk],
    ) -> None:
        """Store stable units and index original chunks by identity."""

        self._units = tuple(units)
        self._parents_by_id = {parent.chunk_id: parent for parent in parents}

    @property
    def units(self) -> tuple[DocumentChunk, ...]:
        """Return the chunks that BM25 and dense retrieval should index."""

        return self._units

    @property
    def parent_chunks(self) -> tuple[DocumentChunk, ...]:
        """Return original chunks in deterministic identity order."""

        return tuple(self._parents_by_id[chunk_id] for chunk_id in sorted(self._parents_by_id))

    def parent_for(self, unit: DocumentChunk) -> DocumentChunk | None:
        """Resolve a derived row or narrative unit to its original chunk."""

        if unit.parent_chunk_id is None:
            return None

        return self._parents_by_id.get(unit.parent_chunk_id)


class ParentAwareRetriever:
    """Expose normal ranked search while retaining parent-table lookup."""

    def __init__(self, *, retriever: RankedRetriever, corpus: TableRetrievalCorpus) -> None:
        """Wrap an existing retriever without changing its ranking behavior."""

        self._retriever = retriever
        self._corpus = corpus

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> tuple[RetrievalResult, ...]:
        """Delegate search to the existing BM25/dense/RRF composition."""

        return self._retriever.search(query, top_k=top_k)

    def parent_for(self, unit: DocumentChunk) -> DocumentChunk | None:
        """Recover the original chunk that contained a derived retrieval unit."""

        return self._corpus.parent_for(unit)


def build_table_retrieval_corpus(
    chunks: Iterable[DocumentChunk],
) -> TableRetrievalCorpus:
    """Replace table-bearing chunks with canonical rows and optional prose."""

    parents = tuple(
        sorted(
            chunks,
            key=lambda chunk: (chunk.document_id, chunk.chunk_index, chunk.chunk_id),
        )
    )
    units: list[DocumentChunk] = []
    seen_rows: set[tuple[str, tuple[str, ...], str, tuple[str, ...], tuple[str, ...]]] = set()

    for parent in parents:
        parsed_tables = _extract_tables(parent.text)

        if not parsed_tables:
            units.append(parent)
            continue

        for table in parsed_tables:
            table_key = (
                parent.document_id,
                parent.source_element_ids,
                _identity_text(table.title),
                tuple(_identity_text(column) for column in table.columns),
            )

            for row in table.rows:
                row_key = (*table_key, tuple(_identity_text(cell) for cell in row))

                if row_key in seen_rows:
                    continue

                seen_rows.add(row_key)
                units.append(_build_row_unit(parent=parent, table=table, row=row))

        narrative = _without_table_spans(parent.text, parsed_tables)

        if _has_narrative_content(narrative):
            units.append(_build_narrative_unit(parent=parent, text=narrative))

    return TableRetrievalCorpus(
        units=sorted(
            units,
            key=lambda unit: (unit.document_id, unit.chunk_index, unit.chunk_id),
        ),
        parents=parents,
    )


def _build_row_unit(
    *,
    parent: DocumentChunk,
    table: _ParsedTable,
    row: tuple[str, ...],
) -> DocumentChunk:
    columns = _resolved_columns(table.columns, column_count=len(row))
    canonical_text = _render_row(title=table.title, columns=columns, row=row)
    identity = _stable_digest(
        parent.document_id,
        "\x1f".join(parent.source_element_ids),
        table.title,
        "\x1f".join(columns),
        "\x1f".join(row),
    )
    row_source_id = f"source:table-row:{identity}"

    return DocumentChunk(
        chunk_id=f"chunk:table-row:{identity}",
        document_id=parent.document_id,
        document_sha256=parent.document_sha256,
        filename=parent.filename,
        chunk_index=parent.chunk_index,
        page_start=parent.page_start,
        page_end=parent.page_end,
        source_element_ids=(row_source_id,),
        text=canonical_text,
        char_count=len(canonical_text),
        token_count=RegexTokenCounter().count(canonical_text),
        block_count=1,
        retrieval_unit_kind="table_row",
        parent_chunk_id=parent.chunk_id,
        parent_source_element_ids=parent.source_element_ids,
    )


def _build_narrative_unit(*, parent: DocumentChunk, text: str) -> DocumentChunk:
    normalized_text = text.strip()
    identity = _stable_digest(parent.chunk_id, "narrative", normalized_text)

    return DocumentChunk(
        chunk_id=f"chunk:narrative:{identity}",
        document_id=parent.document_id,
        document_sha256=parent.document_sha256,
        filename=parent.filename,
        chunk_index=parent.chunk_index,
        page_start=parent.page_start,
        page_end=parent.page_end,
        source_element_ids=(f"source:narrative:{identity}",),
        text=normalized_text,
        char_count=len(normalized_text),
        token_count=RegexTokenCounter().count(normalized_text),
        block_count=1,
        retrieval_unit_kind="narrative",
        parent_chunk_id=parent.chunk_id,
        parent_source_element_ids=parent.source_element_ids,
    )


def _extract_tables(text: str) -> tuple[_ParsedTable, ...]:
    html_tables = tuple(
        _parse_html_table(text, match) for match in _HTML_TABLE_PATTERN.finditer(text)
    )
    markdown_tables = _extract_markdown_tables(text)
    return tuple(sorted((*html_tables, *markdown_tables), key=lambda table: table.start))


def _parse_html_table(text: str, match: re.Match[str]) -> _ParsedTable:
    parser = _HTMLTableParser()
    parser.feed(match.group(0))
    parser.close()
    columns, rows = _separate_headers(parser.rows)
    title = parser.caption or _nearest_table_title(text[: match.start()])
    return _ParsedTable(
        start=match.start(),
        end=match.end(),
        title=title or "Untitled table",
        columns=columns,
        rows=rows,
    )


def _extract_markdown_tables(text: str) -> tuple[_ParsedTable, ...]:
    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    offset = 0

    for line in lines:
        offsets.append(offset)
        offset += len(line)

    tables: list[_ParsedTable] = []
    index = 0

    while index + 1 < len(lines):
        if "|" not in lines[index] or not _MARKDOWN_SEPARATOR_PATTERN.match(lines[index + 1]):
            index += 1
            continue

        end_index = index + 2

        while end_index < len(lines) and "|" in lines[end_index] and lines[end_index].strip():
            end_index += 1

        columns = _parse_markdown_row(lines[index])
        rows = tuple(
            row
            for row in (_parse_markdown_row(line) for line in lines[index + 2 : end_index])
            if row and row != columns
        )
        start = offsets[index]
        end = offsets[end_index] if end_index < len(offsets) else len(text)
        tables.append(
            _ParsedTable(
                start=start,
                end=end,
                title=_nearest_table_title(text[:start]) or "Untitled table",
                columns=columns,
                rows=rows,
            )
        )
        index = end_index

    return tuple(tables)


def _parse_markdown_row(line: str) -> tuple[str, ...]:
    stripped = line.strip().strip("|")
    return tuple(_normalize_cell(cell) for cell in stripped.split("|"))


def _separate_headers(
    rows: tuple[tuple[tuple[str, ...], bool], ...],
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    header_rows: list[tuple[str, ...]] = []
    data_rows: list[tuple[str, ...]] = []

    for cells, is_header in rows:
        if is_header and not data_rows:
            header_rows.append(cells)
        else:
            data_rows.append(cells)

    if header_rows:
        width = max(len(row) for row in header_rows)
        columns = tuple(
            " / ".join(row[index] for row in header_rows if index < len(row) and row[index])
            or f"Column {index + 1}"
            for index in range(width)
        )
    else:
        width = max((len(row) for row in data_rows), default=0)
        columns = tuple(f"Column {index + 1}" for index in range(width))

    normalized_rows = tuple(row for row in data_rows if row and row != columns)
    return columns, normalized_rows


def _resolved_columns(columns: tuple[str, ...], *, column_count: int) -> tuple[str, ...]:
    return tuple(
        columns[index] if index < len(columns) and columns[index] else f"Column {index + 1}"
        for index in range(column_count)
    )


def _render_row(
    *,
    title: str,
    columns: tuple[str, ...],
    row: tuple[str, ...],
) -> str:
    fields = " | ".join(f"{column}: {cell}" for column, cell in zip(columns, row, strict=True))
    return f"Table: {title}\n{fields}"


def _nearest_table_title(prefix: str) -> str:
    lines = prefix.splitlines()

    for line in reversed(lines):
        heading_match = _MARKDOWN_HEADING_PATTERN.match(line)

        if heading_match:
            return _normalize_cell(heading_match.group("title"))

    return ""


def _without_table_spans(text: str, tables: tuple[_ParsedTable, ...]) -> str:
    parts: list[str] = []
    cursor = 0

    for table in tables:
        parts.append(text[cursor : table.start])
        cursor = table.end

    parts.append(text[cursor:])
    return "".join(parts).strip()


def _has_narrative_content(text: str) -> bool:
    without_headings = "\n".join(
        line for line in text.splitlines() if _MARKDOWN_HEADING_PATTERN.match(line) is None
    )
    without_tags = _HTML_TAG_PATTERN.sub(" ", without_headings)
    return bool(_NON_CONTENT_PATTERN.sub("", without_tags))


def _normalize_cell(value: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", value).strip()


def _identity_text(value: str) -> str:
    return _normalize_cell(value).casefold()


def _stable_digest(*values: str) -> str:
    digest = sha256()

    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)

    return digest.hexdigest()[:24]


class _HTMLTableParser(HTMLParser):
    """Extract simple header/data cells from LlamaParse HTML tables."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._in_caption = False
        self._caption_parts: list[str] = []
        self._in_header_section = False
        self._row: list[str] | None = None
        self._row_is_header = False
        self._cell_parts: list[str] | None = None
        self.rows: tuple[tuple[tuple[str, ...], bool], ...] = ()
        self._rows: list[tuple[tuple[str, ...], bool]] = []

    @property
    def caption(self) -> str:
        """Return normalized caption text, if present."""

        return _normalize_cell(" ".join(self._caption_parts))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        normalized_tag = tag.casefold()

        if normalized_tag == "caption":
            self._in_caption = True
        elif normalized_tag == "thead":
            self._in_header_section = True
        elif normalized_tag == "tr":
            self._row = []
            self._row_is_header = self._in_header_section
        elif normalized_tag in {"th", "td"} and self._row is not None:
            self._cell_parts = []

            if normalized_tag == "th":
                self._row_is_header = True
        elif normalized_tag == "br" and self._cell_parts is not None:
            self._cell_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.casefold()

        if normalized_tag == "caption":
            self._in_caption = False
        elif normalized_tag == "thead":
            self._in_header_section = False
        elif normalized_tag in {"th", "td"} and self._row is not None:
            self._row.append(_normalize_cell(" ".join(self._cell_parts or ())))
            self._cell_parts = None
        elif normalized_tag == "tr" and self._row is not None:
            cells = tuple(self._row)

            if any(cells):
                self._rows.append((cells, self._row_is_header))

            self._row = None
            self._row_is_header = False

    def handle_data(self, data: str) -> None:
        if self._in_caption:
            self._caption_parts.append(data)

        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def close(self) -> None:
        super().close()
        self.rows = tuple(self._rows)
