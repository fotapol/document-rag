"""Tests for the document-ingestion web application."""

from dataclasses import dataclass, field

from fastapi.testclient import TestClient

from document_rag.ingestion.chunking import DocumentChunk
from document_rag.ingestion.llamaparse import (
    LlamaParseProcessingError,
    ParsedDocument,
    ParsedPage,
)
from document_rag.rag.errors import RAGNotIndexedError
from document_rag.rag.models import RAGAnswer, SourceCitation
from document_rag.retrieval.models import RetrievalResult
from document_rag.web.app import MAX_UPLOAD_BYTES, create_app


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


@dataclass
class FakeQuestionAnsweringService:
    """Keep a fake session index and answer without external models."""

    indexed_chunks: tuple[DocumentChunk, ...] = ()
    questions: list[str] = field(default_factory=list)

    def index_document(self, chunks: tuple[DocumentChunk, ...]) -> None:
        self.indexed_chunks = chunks

    def answer(self, question: str) -> RAGAnswer:
        if not self.indexed_chunks:
            raise RAGNotIndexedError("Upload and index a PDF before asking a question.")

        self.questions.append(question)
        chunk = self.indexed_chunks[0]
        result = RetrievalResult(chunk=chunk, score=0.031, rank=1)
        return RAGAnswer(
            question=question,
            answer="Revenue was $100 [Source 1].",
            citations=(
                SourceCitation(
                    source_number=1,
                    chunk_id=chunk.chunk_id,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                ),
            ),
            retrieved=(result,),
        )


def test_health() -> None:
    """The process health endpoint should return HTTP 200."""

    client = TestClient(create_app(FakeParser()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_renders_accessible_empty_workspace() -> None:
    """The landing page should explain the workflow before a PDF is uploaded."""

    client = TestClient(create_app(FakeParser()))

    response = client.get("/")

    assert response.status_code == 200
    assert "Ask your document" in response.text
    assert "Upload a PDF to ask a question." in response.text
    assert "Private chat." in response.text
    assert "Answers run locally; LlamaParse processes PDFs." in response.text
    assert "brand-mark" not in response.text
    assert 'href="https://github.com/fotapol"' in response.text
    assert ">fotapol</a>" in response.text
    assert 'aria-label="Application workflow"' in response.text
    assert response.text.count('class="workflow-step"') == 2
    assert 'id="cancel-question"' in response.text
    assert "new AbortController()" in response.text
    assert "questionController.abort()" in response.text
    assert "color-scheme: dark" in response.text
    assert "Model can make mistakes. Check important info." in response.text


def test_parse_and_chunk_pdf_exposes_hidden_document_details() -> None:
    """A valid PDF should offer parsed and indexed details in a closed drawer."""

    question_answering = FakeQuestionAnsweringService()
    client = TestClient(
        create_app(
            FakeParser(),
            question_answering_service=question_answering,
        )
    )

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
    assert "Ask this document" in response.text
    assert "report.pdf is ready for questions." in response.text
    assert "View parsed document" in response.text
    assert '<dialog class="drawer" id="document-details"' in response.text
    assert '<dialog class="drawer" id="document-details" open' not in response.text
    assert "Revenue: $100" in response.text
    assert "sha256:test" in response.text
    assert "Download chunks as JSONL" in response.text
    assert "Source elements" in response.text
    assert "Indexed chunks" in response.text
    assert len(question_answering.indexed_chunks) == 1

    refreshed = client.get("/")

    assert refreshed.status_code == 200
    assert "report.pdf is ready for questions." in refreshed.text


def test_ask_question_displays_clickable_citation_and_hidden_details() -> None:
    """A citation should open supporting text without showing diagnostics by default."""

    question_answering = FakeQuestionAnsweringService()
    client = TestClient(
        create_app(
            FakeParser(),
            question_answering_service=question_answering,
        )
    )
    upload_response = client.post(
        "/documents/parse",
        files={
            "file": (
                "report.pdf",
                b"%PDF-1.7 fake content",
                "application/pdf",
            )
        },
    )

    response = client.post(
        "/questions/ask",
        data={"question": "What was revenue?"},
    )

    assert upload_response.status_code == 200
    assert response.status_code == 200
    assert question_answering.questions == ["What was revenue?"]
    assert "Revenue was $100 [Source 1]." in response.text
    assert ">Sources</h2>" in response.text
    assert "Page 1" in response.text
    assert 'data-open-dialog="source-1"' in response.text
    assert 'id="source-1"' in response.text
    assert 'href="/documents/current.pdf#page=1"' in response.text
    assert "Open original PDF at page 1" in response.text
    assert question_answering.indexed_chunks[0].chunk_id in response.text
    assert "Retrieval score" in response.text
    assert "Technical details" in response.text
    assert 'id="source-1" open' not in response.text
    assert 'id="answer-heading">Answer' in response.text


def test_current_pdf_is_unavailable_before_upload() -> None:
    """The in-memory PDF endpoint should return 404 without a current document."""

    client = TestClient(create_app(FakeParser()))

    response = client.get("/documents/current.pdf")

    assert response.status_code == 404
    assert response.json() == {"detail": "No document is currently available."}


def test_current_pdf_returns_exact_bytes_with_safe_headers() -> None:
    """The source viewer should serve the current PDF inline without caching it."""

    pdf_content = b"%PDF-1.7 exact fake content"
    client = TestClient(
        create_app(
            FakeParser(),
            question_answering_service=FakeQuestionAnsweringService(),
        )
    )
    upload_response = client.post(
        "/documents/parse",
        files={"file": ("financial report.pdf", pdf_content, "application/pdf")},
    )

    response = client.get("/documents/current.pdf")

    assert upload_response.status_code == 200
    assert response.status_code == 200
    assert response.content == pdf_content
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-disposition"] == (
        "inline; filename*=UTF-8''financial%20report.pdf"
    )


def test_failed_replacement_preserves_current_pdf() -> None:
    """A parser failure must not discard the last successfully indexed PDF."""

    original_content = b"%PDF-1.7 original"
    app = create_app(
        FakeParser(),
        question_answering_service=FakeQuestionAnsweringService(),
    )
    client = TestClient(app)
    first_response = client.post(
        "/documents/parse",
        files={"file": ("original.pdf", original_content, "application/pdf")},
    )
    app.state.document_parser = FailingParser()

    failed_response = client.post(
        "/documents/parse",
        files={"file": ("replacement.pdf", b"%PDF-1.7 replacement", "application/pdf")},
    )
    current_pdf_response = client.get("/documents/current.pdf")

    assert first_response.status_code == 200
    assert failed_response.status_code == 502
    assert "original.pdf is ready for questions." in failed_response.text
    assert "replacement.pdf is ready for questions." not in failed_response.text
    assert current_pdf_response.status_code == 200
    assert current_pdf_response.content == original_content
    assert "original.pdf" in current_pdf_response.headers["content-disposition"]


def test_reject_pdf_above_twenty_megabytes() -> None:
    """The application should retain its bounded in-memory upload policy."""

    client = TestClient(create_app(FakeParser()))

    response = client.post(
        "/documents/parse",
        files={
            "file": (
                "oversized.pdf",
                b"%PDF-" + b"0" * (MAX_UPLOAD_BYTES - 4),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 413
    assert "exceeds the 20 MB limit" in response.text


def test_ask_question_before_upload_returns_conflict() -> None:
    """The UI should explain that a PDF must be uploaded first."""

    client = TestClient(
        create_app(
            FakeParser(),
            question_answering_service=FakeQuestionAnsweringService(),
        )
    )

    response = client.post(
        "/questions/ask",
        data={"question": "What was revenue?"},
    )

    assert response.status_code == 409
    assert "Upload and index a PDF" in response.text


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
    assert 'role="alert"' in response.text
