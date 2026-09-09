# Architecture

Document RAG is a local, single-process application for grounded question answering over one
financial PDF at a time. Framework-specific records are converted into typed project models at the
application boundary so ingestion, retrieval, evaluation, and generation remain independently
testable.

## Runtime flow

```text
Browser
  -> FastAPI upload validation (PDF, 20 MB maximum)
  -> LlamaParse cloud API
  -> ParsedDocument and ParsedPage
  -> MarkdownChunker
  -> DocumentChunk with page and source lineage
  -> canonical narrative and table-row retrieval units
  -> BM25+ and BGE dense indexes
  -> deterministic Reciprocal Rank Fusion
  -> parent-aware diversity
  -> top five grounded sources
  -> context-only prompt
  -> pinned Qwen3-1.7B with an unmerged financial LoRA
  -> answer, citations, and optional retrieval diagnostics
```

## Component boundaries

| Component | Responsibility |
|---|---|
| `document_rag.web.app` | HTTP validation, in-memory UI state, and HTML rendering |
| `document_rag.ingestion.llamaparse` | LlamaCloud configuration and PDF parsing |
| `document_rag.ingestion.chunking` | Structure-aware chunks, deterministic IDs, pages, and lineage |
| `document_rag.retrieval.table_units` | Canonical table-row and narrative retrieval children |
| `document_rag.retrieval.bm25` | Deterministic lexical ranking |
| `document_rag.retrieval.dense` | Normalized dense embeddings and cosine ranking |
| `document_rag.retrieval.hybrid` | Reciprocal Rank Fusion over component ranks |
| `document_rag.retrieval.diversity` | Final table-parent diversity selection |
| `document_rag.rag.prompting` | Grounded context, system instruction, and source labels |
| `document_rag.rag.generation` | Lazy deterministic Qwen and PEFT adapter inference |
| `document_rag.rag.service` | In-memory index and end-to-end orchestration |

The domain models do not depend on FastAPI, LlamaParse, Hugging Face, or a database. This keeps
offline unit tests small and lets an external integration be replaced without redefining document
identity or evidence lineage.

## Document and source lineage

`ParsedDocument` contains ordered pages of Markdown. The chunker produces immutable
`DocumentChunk` records that preserve:

- document ID and SHA-256;
- deterministic chunk ID and index;
- inclusive page range;
- source-element IDs;
- text and size statistics; and
- optional parent chunk and parent source identity.

Retrieval results retain the complete chunk rather than copying only its text. Citations can
therefore map an answer back to a page, chunk, and parser-derived source element.

## Table-aware retrieval

Indexing an entire financial table allows query terms from unrelated rows to create a false
combined match. The application expands HTML and Markdown tables into one canonical child chunk per
data row:

```text
Table: Quarterly Revenue Detail
Quarter: 2024 Q1 | Region: North America | Revenue: $220,000
```

Every child repeats the table title and column names. Repeated headers are removed and exact
duplicate rows are indexed only once. Each row receives content-derived chunk and source IDs,
preserves the original page range, and points back to the parent table chunk and its source
lineage.

If prose and a table share an original chunk, the prose becomes a separate narrative child.
`ParentAwareRetriever.parent_for()` can recover the complete original chunk when full-table
context is needed.

## Retrieval and ranking

The uploaded-document index is rebuilt in memory after each successful upload:

1. **BM25+** uses deterministic financial-token normalization. Common question boilerplate and
   duplicate query terms are removed while currencies, percentages, years, quarters, and numeric
   values are preserved.
2. **Dense retrieval** uses normalized vectors from pinned
   `BAAI/bge-small-en-v1.5` and exact cosine similarity.
3. **RRF** combines ranks rather than incomparable raw lexical and dense scores:

   ```text
   score(d) = sum(1 / (rrf_k + component_rank(d)))
   ```

4. **Parent diversity** greedily keeps RRF order while limiting sibling rows from one logical table
   source. Narrative chunks are not capped.

Stable chunk IDs break equal-score ties. Component ranks, RRF scores, pages, and lineage remain
available in the opt-in source drawer.

The standalone frozen BM25, dense, and hybrid benchmark implementations are not changed by the
small-document BM25+ or diversity choices used in the web application.

## Grounded generation

The prompt contains:

1. a system instruction requiring context-only answers and explicit refusal when evidence is
   insufficient;
2. retrieved chunks labeled with `[Source N | page ... | chunk_id ...]`; and
3. the current user question.

Displayed answer history is never added to this prompt.

`QwenLoraGenerator` lazily loads the tokenizer from the adapter repository, the separately pinned
Qwen base model, and the pinned PEFT adapter. Adapter weights are not merged into the base model.
Generation uses `do_sample=False`, disables Qwen thinking in the chat template, and fails rather
than silently truncating an oversized grounded prompt.

## Application state

One FastAPI process stores:

- the most recently successful parsed document;
- its original PDF bytes;
- the current chunks and retrieval index; and
- up to ten display-only answers.

A failed replacement upload leaves the working document untouched. A successful replacement clears
old answers. **Clear answers** retains the PDF and index; **Remove document** clears the PDF, index,
and answers but does not unload model weights.

The PDF endpoint returns the in-memory bytes with `Cache-Control: no-store` so page citations can
open the original document. It is intentionally unauthenticated for localhost use.

## Privacy and trust boundary

PDF bytes cross the local boundary when sent to LlamaParse. After parsing, retrieval and generation
run locally. No application persistence is implemented, but lack of persistence is not equivalent
to multi-user privacy: every client connected to the same process shares its current state.

This server must not be exposed publicly without authentication, per-user state isolation, upload
hardening, authorization on the PDF endpoint, and an explicit retention policy.

## Deliberate non-goals

The current release does not include:

- a vector database or Qdrant;
- reranking;
- multi-user persistence;
- conversational model memory;
- multiple embedding models;
- distributed inference; or
- production deployment security.
