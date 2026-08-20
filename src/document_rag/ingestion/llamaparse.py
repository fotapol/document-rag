"""LlamaParse integration for document ingestion."""

from __future__ import annotations

import os
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal, cast

from llama_cloud import LlamaCloud

type ParseTier = Literal[
    "fast",
    "cost_effective",
    "agentic",
    "agentic_plus",
]

SUPPORTED_PARSE_TIERS = frozenset(
    {
        "fast",
        "cost_effective",
        "agentic",
        "agentic_plus",
    }
)
DEFAULT_PARSE_TIER: ParseTier = "agentic"
DEFAULT_PARSE_VERSION = "latest"
DEFAULT_TIMEOUT_SECONDS = 900.0


class LlamaParseConfigurationError(RuntimeError):
    """Raised when the LlamaParse client is not configured."""


class LlamaParseProcessingError(RuntimeError):
    """Raised when LlamaParse cannot produce a complete Markdown result."""


def parse_tier(value: str) -> ParseTier:
    """Validate a dynamically configured LlamaParse tier."""

    if value not in SUPPORTED_PARSE_TIERS:
        supported = ", ".join(sorted(SUPPORTED_PARSE_TIERS))
        raise LlamaParseConfigurationError(
            f"Unsupported LlamaParse tier {value!r}; expected one of: {supported}."
        )

    return cast(ParseTier, value)


@dataclass(frozen=True, slots=True)
class ParsedPage:
    """Page Markdown and optional IDs aligned with its structural blocks."""

    page_number: int
    markdown: str
    source_element_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Deterministic document identity and page-level Markdown."""

    document_id: str
    filename: str
    sha256: str
    pages: tuple[ParsedPage, ...]

    @property
    def markdown(self) -> str:
        """Join page Markdown while preserving explicit page boundaries."""

        return "\n\n".join(
            (f"<!-- page: {page.page_number} -->\n{page.markdown}") for page in self.pages
        )


class LlamaParseService:
    """Parse uploaded PDF bytes with the official Llama Cloud SDK."""

    def __init__(
        self,
        *,
        api_key: str,
        tier: str = DEFAULT_PARSE_TIER,
        version: str = DEFAULT_PARSE_VERSION,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client: Any | None = None,
    ) -> None:
        """Create a reusable parser client with explicit parse settings."""

        if not api_key.strip():
            raise LlamaParseConfigurationError("LLAMA_CLOUD_API_KEY is not configured.")

        self._tier = parse_tier(tier)
        self._version = version
        self._timeout_seconds = timeout_seconds
        self._client = client or LlamaCloud(api_key=api_key)

    @classmethod
    def from_environment(cls) -> LlamaParseService:
        """Create the service from environment variables."""

        return cls(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY", ""),
            tier=os.getenv(
                "LLAMA_PARSE_TIER",
                DEFAULT_PARSE_TIER,
            ),
            version=os.getenv(
                "LLAMA_PARSE_VERSION",
                DEFAULT_PARSE_VERSION,
            ),
            timeout_seconds=float(
                os.getenv(
                    "LLAMA_PARSE_TIMEOUT_SECONDS",
                    str(DEFAULT_TIMEOUT_SECONDS),
                )
            ),
        )

    def parse_pdf(
        self,
        *,
        filename: str,
        content_type: str,
        content: bytes,
    ) -> ParsedDocument:
        """Parse one PDF and reject partial or empty page results."""

        result = self._client.parsing.parse(
            tier=self._tier,
            version=self._version,
            upload_file=(
                filename,
                content,
                content_type,
            ),
            expand=["markdown"],
            timeout=self._timeout_seconds,
            client_name="document-rag",
        )

        if result.markdown is None:
            raise LlamaParseProcessingError("LlamaParse returned no Markdown result.")

        pages: list[ParsedPage] = []
        failures: list[str] = []

        for page in result.markdown.pages:
            if page.success is True:
                markdown = page.markdown.strip()

                if markdown:
                    pages.append(
                        ParsedPage(
                            page_number=page.page_number,
                            markdown=markdown,
                        )
                    )
                continue

            if page.success is False:
                failures.append(f"page {page.page_number}: {page.error}")
                continue

            failures.append(f"page {page.page_number}: {getattr(page, 'error', 'unknown error')}")

        if failures:
            raise LlamaParseProcessingError(
                "LlamaParse returned failed pages: " + "; ".join(failures)
            )

        if not pages:
            raise LlamaParseProcessingError("LlamaParse returned only empty pages.")

        digest = sha256(content).hexdigest()

        return ParsedDocument(
            document_id=f"sha256:{digest}",
            filename=filename,
            sha256=digest,
            pages=tuple(pages),
        )
