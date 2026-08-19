"""Tests for the document-ingestion web application."""

from dataclasses import dataclass

from fastapi.testclient import TestClient

from document_rag.ingestion.llamaparse import (
    LlamaParseProcessingError,
    ParsedDocument,
    ParsedPage,
)
from document_rag.web.app import create_app


@dataclass
class FakeParser:
    """Return a deterministic parsed document without external API calls."""

    def parse_pdf(
        self,
        *,
        filename: str,
        content_type: str,
        content: bytes,
    ) -> ParsedDocument:
        """Return one fake Markdown page."""

        del content_type, content

        return ParsedDocument(
            document_id="sha256:test",
            filename=filename,
            sha256="test",
            pages=(
                ParsedPage(
                    page_number=1,
                    markdown="# Parsed page\n\nRevenue: $100",
                ),
            ),
        )


@dataclass
class FailingParser:
    """Simulate a known external parsing failure."""

    def parse_pdf(
        self,
        *,
        filename: str,
        content_type: str,
        content: bytes,
    ) -> ParsedDocument:
        """Raise the same error type as the real parsing service."""

        del filename, content_type, content
        raise LlamaParseProcessingError("Parser failed.")


def test_health() -> None:
    """The process health endpoint should return HTTP 200."""

    client = TestClient(create_app(FakeParser()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_parse_and_chunk_pdf() -> None:
    """A valid PDF should render Markdown, chunks, and JSONL download."""

    client = TestClient(create_app(FakeParser()))

    response = client.post(
        "/documents/parse",
        files={
            "file": (
                "report.pdf",
                b"%PDF-1.7 fake content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200
    assert "report.pdf" in response.text
    assert "Revenue: $100" in response.text
    assert "sha256:test" in response.text
    assert "Retrieval chunks" in response.text
    assert "Download chunks as JSONL" in response.text
    assert "Tokens" in response.text
    assert "chunk:" in response.text


def test_reject_non_pdf_extension() -> None:
    """A non-PDF filename should be rejected before parsing."""

    client = TestClient(create_app(FakeParser()))

    response = client.post(
        "/documents/parse",
        files={
            "file": (
                "report.txt",
                b"%PDF-1.7 fake content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert "Only PDF files are accepted." in response.text


def test_reject_invalid_pdf_signature() -> None:
    """A spoofed PDF content type should not bypass signature validation."""

    client = TestClient(create_app(FakeParser()))

    response = client.post(
        "/documents/parse",
        files={
            "file": (
                "report.pdf",
                b"not a pdf",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert "not a valid PDF" in response.text


def test_render_known_parser_failure() -> None:
    """A known parser error should be shown without exposing a traceback."""

    client = TestClient(create_app(FailingParser()))

    response = client.post(
        "/documents/parse",
        files={
            "file": (
                "report.pdf",
                b"%PDF-1.7 fake content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 502
    assert "Parser failed." in response.text
