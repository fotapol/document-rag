"""FastAPI document-ingestion web application."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Protocol, cast
from urllib.parse import quote

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
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
from document_rag.rag.models import ModelStatus, RAGAnswer
from document_rag.rag.service import RAGService, build_default_rag_service

LOGGER = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_ANSWER_HISTORY = 10
PDF_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
}

TEMPLATE_DIRECTORY = Path(__file__).with_name("templates")
TEMPLATES = Jinja2Templates(directory=TEMPLATE_DIRECTORY)

MODEL_STATUS_MESSAGES: dict[ModelStatus, str] = {
    "not_loaded": "The local model will load when you ask the first question.",
    "loading": "Loading the local model. The first answer can take a few minutes.",
    "ready": "The local model is ready.",
    "error": "The local model could not start. Check the server logs.",
}


@dataclass(frozen=True, slots=True)
class AnswerHistoryEntry:
    """One display-only question and answer retained for the current document."""

    entry_id: int
    answer: RAGAnswer


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

    def clear_document(self) -> None:
        """Discard the current in-memory retrieval index."""


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
    app.state.current_pdf_content = None
    app.state.current_pdf_filename = None
    app.state.answer_history = ()
    app.state.next_answer_id = 1

    @app.get("/health")
    async def health() -> JSONResponse:
        """Return a lightweight process health check."""

        return JSONResponse({"status": "ok"})

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        """Render the PDF upload page."""

        document, chunks = get_current_document(request)
        return render_result(
            request,
            document=document,
            chunks=chunks,
            chunks_jsonl=chunks_to_jsonl(chunks),
            status_code=200,
        )

    @app.get("/documents/current.pdf", response_class=Response)
    async def current_pdf(request: Request) -> Response:
        """Return the current session PDF for page-level citation links."""

        content = cast(bytes | None, request.app.state.current_pdf_content)
        filename = cast(str | None, request.app.state.current_pdf_filename)

        if content is None or filename is None:
            raise HTTPException(status_code=404, detail="No document is currently available.")

        encoded_filename = quote(filename, safe="")
        return Response(
            content=content,
            media_type="application/pdf",
            headers={
                "Cache-Control": "no-store",
                "Content-Disposition": f"inline; filename*=UTF-8''{encoded_filename}",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/model/status", response_class=JSONResponse)
    async def model_status(request: Request) -> JSONResponse:
        """Report lazy local-model readiness without triggering model loading."""

        status = get_model_status(request)
        return JSONResponse(
            {
                "status": status,
                "message": MODEL_STATUS_MESSAGES[status],
            },
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/documents/parse", response_class=HTMLResponse)
    async def parse_document(
        request: Request,
        file: UploadFile,
    ) -> HTMLResponse:
        """Validate, parse, chunk, and render one uploaded PDF."""

        current_document, current_chunks = get_current_document(request)
        current_chunks_jsonl = chunks_to_jsonl(current_chunks)
        error = validate_upload_metadata(file)

        if error is not None:
            return render_result(
                request,
                document=current_document,
                chunks=current_chunks,
                chunks_jsonl=current_chunks_jsonl,
                error=error,
                status_code=400,
            )

        content = await read_limited_upload(file)

        if content is None:
            return render_result(
                request,
                document=current_document,
                chunks=current_chunks,
                chunks_jsonl=current_chunks_jsonl,
                error=(
                    f"The uploaded file exceeds the {MAX_UPLOAD_BYTES // 1024 // 1024} MB limit."
                ),
                status_code=413,
            )

        if not content.startswith(b"%PDF-"):
            return render_result(
                request,
                document=current_document,
                chunks=current_chunks,
                chunks_jsonl=current_chunks_jsonl,
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
                document=current_document,
                chunks=current_chunks,
                chunks_jsonl=current_chunks_jsonl,
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
                document=current_document,
                chunks=current_chunks,
                chunks_jsonl=current_chunks_jsonl,
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
                document=current_document,
                chunks=current_chunks,
                chunks_jsonl=current_chunks_jsonl,
                error="Unexpected document processing failure.",
                status_code=502,
            )

        request.app.state.current_document = document
        request.app.state.current_chunks = chunks
        request.app.state.current_pdf_content = content
        request.app.state.current_pdf_filename = document.filename
        request.app.state.answer_history = ()
        request.app.state.next_answer_id = 1

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

        if document is None:
            return render_result(
                request,
                error="Upload and index a PDF before asking a question.",
                question=normalized_question,
                status_code=409,
            )

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
                document=document,
                chunks=chunks,
                chunks_jsonl=chunks_to_jsonl(chunks),
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

        append_answer_history(request, answer)
        return render_result(
            request,
            document=document,
            chunks=chunks,
            chunks_jsonl=chunks_to_jsonl(chunks),
            status_code=200,
        )

    @app.post("/answers/clear", response_class=HTMLResponse)
    async def clear_answers(request: Request) -> HTMLResponse:
        """Clear display-only answer history while retaining the current document."""

        request.app.state.answer_history = ()
        request.app.state.next_answer_id = 1
        document, chunks = get_current_document(request)
        return render_result(
            request,
            document=document,
            chunks=chunks,
            chunks_jsonl=chunks_to_jsonl(chunks),
            status_code=200,
        )

    @app.post("/documents/remove", response_class=HTMLResponse)
    async def remove_document(request: Request) -> HTMLResponse:
        """Remove the document, retrieval index, PDF bytes, and display history."""

        service = cast(
            QuestionAnsweringService | None,
            request.app.state.question_answering_service,
        )
        if service is not None:
            await run_in_threadpool(service.clear_document)

        request.app.state.current_document = None
        request.app.state.current_chunks = ()
        request.app.state.current_pdf_content = None
        request.app.state.current_pdf_filename = None
        request.app.state.answer_history = ()
        request.app.state.next_answer_id = 1
        return render_result(request, status_code=200)

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


def get_model_status(request: Request) -> ModelStatus:
    """Return service model readiness without creating the real service."""

    service = request.app.state.question_answering_service
    if service is None:
        return "not_loaded"

    status = getattr(service, "model_status", "ready")
    if status in MODEL_STATUS_MESSAGES:
        return cast(ModelStatus, status)
    return "ready"


def append_answer_history(request: Request, answer: RAGAnswer) -> None:
    """Append one display entry and retain only the most recent answers."""

    history = cast(
        tuple[AnswerHistoryEntry, ...],
        request.app.state.answer_history,
    )
    entry_id = cast(int, request.app.state.next_answer_id)
    history = (*history, AnswerHistoryEntry(entry_id=entry_id, answer=answer))
    request.app.state.answer_history = history[-MAX_ANSWER_HISTORY:]
    request.app.state.next_answer_id = entry_id + 1


def render_result(
    request: Request,
    *,
    document: ParsedDocument | None = None,
    chunks: tuple[DocumentChunk, ...] = (),
    chunks_jsonl: str = "",
    error: str | None = None,
    question: str = "",
    status_code: int,
) -> HTMLResponse:
    """Render the upload page, processing result, or user-facing error."""

    question_answering = request.app.state.question_answering_service
    technical_configuration: tuple[tuple[str, str], ...] = ()
    answer_history = cast(
        tuple[AnswerHistoryEntry, ...],
        request.app.state.answer_history,
    )
    model_status = get_model_status(request)

    if isinstance(question_answering, RAGService):
        technical_configuration = question_answering.technical_configuration

    return TEMPLATES.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "document": document,
            "chunks": chunks,
            "chunks_jsonl": chunks_jsonl,
            "error": error,
            "question": question,
            "answer_history": tuple(reversed(answer_history)),
            "max_upload_mb": MAX_UPLOAD_BYTES // 1024 // 1024,
            "model_status": model_status,
            "model_status_message": MODEL_STATUS_MESSAGES[model_status],
            "technical_configuration": technical_configuration,
        },
        status_code=status_code,
    )


app = create_app()
