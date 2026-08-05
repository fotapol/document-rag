"""FastAPI document-ingestion web application."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, cast

from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from document_rag.ingestion.llamaparse import (
    LlamaParseConfigurationError,
    LlamaParseProcessingError,
    LlamaParseService,
    ParsedDocument,
)

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


def create_app(
    document_parser: DocumentParser | None = None,
) -> FastAPI:
    """Create the application and allow parser injection in tests."""

    app = FastAPI(
        title="Document RAG",
        version="0.1.0",
    )
    app.state.document_parser = document_parser

    @app.get("/health")
    async def health() -> JSONResponse:
        """Return a lightweight process health check."""

        return JSONResponse({"status": "ok"})

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        """Render the PDF upload page."""

        return TEMPLATES.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "document": None,
                "error": None,
                "max_upload_mb": MAX_UPLOAD_BYTES // 1024 // 1024,
            },
        )

    @app.post("/documents/parse", response_class=HTMLResponse)
    async def parse_document(
        request: Request,
        file: UploadFile,
    ) -> HTMLResponse:
        """Validate, parse, and render one uploaded PDF."""

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
        except LlamaParseConfigurationError as exc:
            return render_result(
                request,
                error=str(exc),
                status_code=503,
            )
        except LlamaParseProcessingError as exc:
            LOGGER.warning(
                "LlamaParse could not process %s: %s",
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
                "Unexpected document parsing failure for %s",
                file.filename,
            )
            return render_result(
                request,
                error="Unexpected document parsing failure.",
                status_code=502,
            )

        return render_result(
            request,
            document=document,
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


def render_result(
    request: Request,
    *,
    document: ParsedDocument | None = None,
    error: str | None = None,
    status_code: int,
) -> HTMLResponse:
    """Render a successful parse result or a user-facing error."""

    return TEMPLATES.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "document": document,
            "error": error,
            "max_upload_mb": MAX_UPLOAD_BYTES // 1024 // 1024,
        },
        status_code=status_code,
    )


app = create_app()
