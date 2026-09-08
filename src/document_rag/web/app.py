"""FastAPI document-ingestion web application."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Protocol, cast

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from document_rag.ingestion.chunking import (
    DocumentChunk,
    DocumentChunkingError,
    MarkdownChunker,
    chunks_to_jsonl,
)
from document_rag.ingestion.llamaparse import (
    LlamaParseConfigurationError,
    LlamaParseProcessingError,
    LlamaParseService,
    ParsedDocument,
)
from document_rag.rag.errors import (
    RAGConfigurationError,
    RAGGenerationError,
    RAGIndexingError,
    RAGNotIndexedError,
)
from document_rag.rag.models import RAGAnswer
from document_rag.rag.service import build_default_rag_service

LOGGER = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
PDF_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
}

TEMPLATE_DIRECTORY = Path(__file__).with_name("templates")
TEMPLATES = Jinja2Templates(directory=TEMPLATE_DIRECTORY)


class DocumentParser(Protocol):
    """Interface required by the web layer for document parsing."""

    def parse_pdf(
        self,
        *,
        filename: str,
        content_type: str,
        content: bytes,
    ) -> ParsedDocument:
        """Parse one validated PDF upload."""


class DocumentChunker(Protocol):
    """Interface required by the web layer for document chunking."""

    def chunk(
        self,
        document: ParsedDocument,
    ) -> tuple[DocumentChunk, ...]:
        """Convert one parsed document into normalized chunks."""


class QuestionAnsweringService(Protocol):
    """Application-layer operations used by the web interface."""

    def index_document(self, chunks: tuple[DocumentChunk, ...]) -> None:
        """Replace the current in-memory document index."""

    def answer(self, question: str) -> RAGAnswer:
        """Answer one question from the current in-memory index."""


def create_app(
    document_parser: DocumentParser | None = None,
    document_chunker: DocumentChunker | None = None,
    question_answering_service: QuestionAnsweringService | None = None,
) -> FastAPI:
    """Create the application and allow dependency injection in tests."""

    app = FastAPI(
        title="Document RAG",
        version="0.2.0",
    )
    app.state.document_parser = document_parser
    app.state.document_chunker = document_chunker or MarkdownChunker()
    app.state.question_answering_service = question_answering_service
    app.state.current_document = None
    app.state.current_chunks = ()

    @app.get("/health")
    async def health() -> JSONResponse:
        """Return a lightweight process health check."""

        return JSONResponse({"status": "ok"})

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        """Render the PDF upload page."""

        return render_result(
            request,
            status_code=200,
        )

    @app.post("/documents/parse", response_class=HTMLResponse)
    async def parse_document(
        request: Request,
        file: UploadFile,
    ) -> HTMLResponse:
        """Validate, parse, chunk, and render one uploaded PDF."""

        error = validate_upload_metadata(file)

        if error is not None:
            return render_result(
                request,
                error=error,
                status_code=400,
            )

        content = await read_limited_upload(file)

        if content is None:
            return render_result(
                request,
                error=(
                    f"The uploaded file exceeds the {MAX_UPLOAD_BYTES // 1024 // 1024} MB limit."
                ),
                status_code=413,
            )

        if not content.startswith(b"%PDF-"):
            return render_result(
                request,
                error="The uploaded file is not a valid PDF.",
                status_code=400,
            )

        try:
            parser = get_document_parser(request)
            document = await run_in_threadpool(
                parser.parse_pdf,
                filename=file.filename or "document.pdf",
                content_type=file.content_type or "application/pdf",
                content=content,
            )

            chunker = get_document_chunker(request)
            chunks = chunker.chunk(document)
            question_answering = get_question_answering_service(request)
            await run_in_threadpool(
                question_answering.index_document,
                chunks,
            )
            chunks_jsonl = chunks_to_jsonl(chunks)
        except (
            LlamaParseConfigurationError,
            RAGConfigurationError,
        ) as exc:
            return render_result(
                request,
                error=str(exc),
                status_code=503,
            )
        except (
            LlamaParseProcessingError,
            DocumentChunkingError,
            RAGIndexingError,
        ) as exc:
            LOGGER.warning(
                "Document processing failed for %s: %s",
                file.filename,
                exc,
            )
            return render_result(
                request,
                error=str(exc),
                status_code=502,
            )
        except Exception:
            LOGGER.exception(
                "Unexpected document processing failure for %s",
                file.filename,
            )
            return render_result(
                request,
                error="Unexpected document processing failure.",
                status_code=502,
            )

        request.app.state.current_document = document
        request.app.state.current_chunks = chunks

        return render_result(
            request,
            document=document,
            chunks=chunks,
            chunks_jsonl=chunks_jsonl,
            status_code=200,
        )

    @app.post("/questions/ask", response_class=HTMLResponse)
    async def ask_question(
        request: Request,
        question: Annotated[str, Form()],
    ) -> HTMLResponse:
        """Answer one question from the current application-session index."""

        document, chunks = get_current_document(request)
        normalized_question = question.strip()

        if not normalized_question:
            return render_result(
                request,
                document=document,
                chunks=chunks,
                chunks_jsonl=chunks_to_jsonl(chunks),
                error="Enter a question about the uploaded document.",
                question=question,
                status_code=400,
            )

        try:
            question_answering = get_question_answering_service(request)
            answer = await run_in_threadpool(
                question_answering.answer,
                normalized_question,
            )
        except RAGNotIndexedError as exc:
            return render_result(
                request,
                error=str(exc),
                question=normalized_question,
                status_code=409,
            )
        except RAGConfigurationError as exc:
            return render_result(
                request,
                document=document,
                chunks=chunks,
                chunks_jsonl=chunks_to_jsonl(chunks),
                error=str(exc),
                question=normalized_question,
                status_code=503,
            )
        except RAGGenerationError as exc:
            LOGGER.warning("Answer generation failed: %s", exc, exc_info=True)
            return render_result(
                request,
                document=document,
                chunks=chunks,
                chunks_jsonl=chunks_to_jsonl(chunks),
                error=str(exc),
                question=normalized_question,
                status_code=502,
            )
        except Exception:
            LOGGER.exception("Unexpected question-answering failure")
            return render_result(
                request,
                document=document,
                chunks=chunks,
                chunks_jsonl=chunks_to_jsonl(chunks),
                error="Unexpected question-answering failure.",
                question=normalized_question,
                status_code=502,
            )

        return render_result(
            request,
            document=document,
            chunks=chunks,
            chunks_jsonl=chunks_to_jsonl(chunks),
            question=normalized_question,
            answer=answer,
            status_code=200,
        )

    return app


def validate_upload_metadata(file: UploadFile) -> str | None:
    """Validate filename extension and declared media type."""

    filename = file.filename or ""

    if Path(filename).suffix.lower() != ".pdf":
        return "Only PDF files are accepted."

    if file.content_type not in PDF_CONTENT_TYPES:
        return "The upload must use the application/pdf content type."

    return None


async def read_limited_upload(file: UploadFile) -> bytes | None:
    """Read at most one byte beyond the configured upload limit."""

    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()

    if len(content) > MAX_UPLOAD_BYTES:
        return None

    return content


def get_document_parser(request: Request) -> DocumentParser:
    """Return the injected parser or lazily create the real service."""

    parser = cast(
        DocumentParser | None,
        request.app.state.document_parser,
    )

    if parser is None:
        parser = LlamaParseService.from_environment()
        request.app.state.document_parser = parser

    return parser


def get_document_chunker(request: Request) -> DocumentChunker:
    """Return the configured deterministic document chunker."""

    return cast(
        DocumentChunker,
        request.app.state.document_chunker,
    )


def get_question_answering_service(request: Request) -> QuestionAnsweringService:
    """Return the injected service or lazily create the real RAG pipeline."""

    service = cast(
        QuestionAnsweringService | None,
        request.app.state.question_answering_service,
    )

    if service is None:
        service = build_default_rag_service()
        request.app.state.question_answering_service = service

    return service


def get_current_document(
    request: Request,
) -> tuple[ParsedDocument | None, tuple[DocumentChunk, ...]]:
    """Return the document and chunks stored for the current server session."""

    document = cast(
        ParsedDocument | None,
        request.app.state.current_document,
    )
    chunks = cast(
        tuple[DocumentChunk, ...],
        request.app.state.current_chunks,
    )
    return document, chunks


def render_result(
    request: Request,
    *,
    document: ParsedDocument | None = None,
    chunks: tuple[DocumentChunk, ...] = (),
    chunks_jsonl: str = "",
    error: str | None = None,
    question: str = "",
    answer: RAGAnswer | None = None,
    status_code: int,
) -> HTMLResponse:
    """Render the upload page, processing result, or user-facing error."""

    return TEMPLATES.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "document": document,
            "chunks": chunks,
            "chunks_jsonl": chunks_jsonl,
            "error": error,
            "question": question,
            "answer": answer,
            "max_upload_mb": MAX_UPLOAD_BYTES // 1024 // 1024,
        },
        status_code=status_code,
    )


app = create_app()
