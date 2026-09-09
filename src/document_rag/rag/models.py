"""Immutable records exchanged by the RAG application layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from document_rag.retrieval.models import RetrievalResult

type ChatRole = Literal["system", "user"]
type ModelStatus = Literal["not_loaded", "loading", "ready", "error"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One model-ready chat message."""

    role: ChatRole
    content: str

    def to_record(self) -> dict[str, str]:
        """Return the mapping expected by Hugging Face chat templates."""

        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class GroundedPrompt:
    """A complete system-and-user prompt built from retrieved evidence."""

    messages: tuple[ChatMessage, ...]

    def to_messages(self) -> list[dict[str, str]]:
        """Return mutable mappings accepted by tokenizer chat templates."""

        return [message.to_record() for message in self.messages]


@dataclass(frozen=True, slots=True)
class SourceCitation:
    """Stable UI citation for one retrieved context item."""

    source_number: int
    chunk_id: str
    page_start: int
    page_end: int

    @property
    def page_label(self) -> str:
        """Render a single page or inclusive page range."""

        if self.page_start == self.page_end:
            return str(self.page_start)

        return f"{self.page_start}-{self.page_end}"


@dataclass(frozen=True, slots=True)
class RAGAnswer:
    """Grounded answer, citations, and optional retrieval diagnostics."""

    question: str
    answer: str
    citations: tuple[SourceCitation, ...]
    retrieved: tuple[RetrievalResult, ...]
