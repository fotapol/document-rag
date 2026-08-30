"""Tests for deterministic grounded prompt construction."""

from document_rag.ingestion.chunking import DocumentChunk
from document_rag.rag.prompting import (
    SYSTEM_INSTRUCTION,
    UNSUPPORTED_ANSWER,
    build_citations,
    build_grounded_prompt,
)
from document_rag.retrieval.models import RetrievalResult


def test_prompt_contains_system_instruction_context_lineage_and_question() -> None:
    """The model prompt should keep instructions separate from retrieved evidence."""

    result = RetrievalResult(
        chunk=_chunk(
            chunk_id="chunk:revenue",
            page_start=7,
            page_end=8,
            text="Revenue increased to $14.1 million.",
        ),
        score=0.9,
        rank=1,
    )

    prompt = build_grounded_prompt(
        question="What was reported revenue?",
        results=(result,),
    )

    assert prompt.messages[0].role == "system"
    assert prompt.messages[0].content == SYSTEM_INSTRUCTION
    assert "Use only the retrieved context" in SYSTEM_INSTRUCTION
    assert UNSUPPORTED_ANSWER in SYSTEM_INSTRUCTION
    assert prompt.messages[1].role == "user"
    assert "[Source 1 | page 7-8 | chunk_id chunk:revenue]" in prompt.messages[1].content
    assert "Revenue increased to $14.1 million." in prompt.messages[1].content
    assert "Question:\nWhat was reported revenue?" in prompt.messages[1].content


def test_citations_preserve_ranked_page_and_chunk_identity() -> None:
    """Every context item should produce a stable page-and-chunk citation."""

    results = (
        RetrievalResult(
            chunk=_chunk(
                chunk_id="chunk:first",
                page_start=2,
                page_end=2,
                text="First",
            ),
            score=0.8,
            rank=1,
        ),
        RetrievalResult(
            chunk=_chunk(
                chunk_id="chunk:second",
                page_start=4,
                page_end=5,
                text="Second",
            ),
            score=0.7,
            rank=2,
        ),
    )

    citations = build_citations(results)

    assert tuple(citation.chunk_id for citation in citations) == (
        "chunk:first",
        "chunk:second",
    )
    assert citations[0].page_label == "2"
    assert citations[1].page_label == "4-5"


def _chunk(
    *,
    chunk_id: str,
    page_start: int,
    page_end: int,
    text: str,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="document:test",
        document_sha256="test",
        filename="report.pdf",
        chunk_index=page_start,
        page_start=page_start,
        page_end=page_end,
        source_element_ids=(f"element:{chunk_id}",),
        text=text,
        char_count=len(text),
        token_count=len(text.split()),
        block_count=1,
    )
