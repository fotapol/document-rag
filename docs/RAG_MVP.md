# End-to-end RAG MVP

The web application now executes the first complete document question-answering flow:

```text
PDF upload
-> LlamaParse Markdown
-> structure-aware DocumentChunk records
-> in-memory BM25 + BGE dense indexes
-> deterministic Reciprocal Rank Fusion
-> top five grounded context chunks
-> pinned Qwen3-1.7B + pinned financial LoRA
-> answer with page and chunk citations
```

The application reuses the existing ingestion and retrieval implementations. It does not use a
vector database, reranker, persistent index, or merged adapter. The most recently uploaded PDF
replaces the application-process index; restarting the server clears it. This is deliberately a
single-session MVP rather than multi-user persistence.

## Architecture

- `document_rag.web.app` validates uploads, renders HTML, and delegates work.
- `RAGService` owns the current in-memory index and orchestrates retrieval, prompting, and
  generation.
- `InMemoryHybridIndexFactory` composes the existing `BM25Retriever`, `DenseRetriever`,
  `SentenceTransformerEmbedder`, and `ReciprocalRankFusionRetriever` classes.
- `build_grounded_prompt` labels every context item with its page and deterministic chunk ID. Its
  system instruction requires context-only answers and an explicit unsupported-answer response.
- `QwenLoraGenerator` lazily loads the pinned base model and attaches the published PEFT adapter
  with `PeftModel.from_pretrained`. It never calls `merge_and_unload` and uses `do_sample=False`.

The dense embedding model loads during the first successful PDF upload. The Qwen base model and
LoRA adapter load only when the first question is submitted.

## Configuration

All settings are optional except `LLAMA_CLOUD_API_KEY` for real PDF parsing.

| Environment variable | Default |
|---|---|
| `LLAMA_CLOUD_API_KEY` | No default |
| `DOCUMENT_RAG_TOP_K` | `5` |
| `DOCUMENT_RAG_CANDIDATE_K` | `20` per retriever |
| `DOCUMENT_RAG_RRF_K` | `60` |
| `DOCUMENT_RAG_EMBEDDING_MODEL_ID` | `BAAI/bge-small-en-v1.5` |
| `DOCUMENT_RAG_EMBEDDING_MODEL_REVISION` | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| `DOCUMENT_RAG_EMBEDDING_DEVICE` | `auto` |
| `DOCUMENT_RAG_DENSE_BATCH_SIZE` | `32` |
| `DOCUMENT_RAG_BASE_MODEL_ID` | `Qwen/Qwen3-1.7B` |
| `DOCUMENT_RAG_BASE_MODEL_REVISION` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| `DOCUMENT_RAG_ADAPTER_MODEL_ID` | `fotapol/qwen3-1.7b-financial-qa-lora` |
| `DOCUMENT_RAG_ADAPTER_MODEL_REVISION` | `8433fcf8142e3db64f38df1f6eeaa38bf2e92651` |
| `DOCUMENT_RAG_GENERATION_DEVICE_MAP` | `auto` |
| `DOCUMENT_RAG_MAX_INPUT_TOKENS` | `4096` |
| `DOCUMENT_RAG_MAX_NEW_TOKENS` | `128` |

`DOCUMENT_RAG_CANDIDATE_K` must be at least `DOCUMENT_RAG_TOP_K`. The adapter is public, so a
Hugging Face token is not required unless the repository visibility changes. Prompts exceeding
`DOCUMENT_RAG_MAX_INPUT_TOKENS` fail explicitly instead of silently truncating the system
instruction, evidence, or question; reduce `DOCUMENT_RAG_TOP_K` or raise the input limit when
needed.

## Run command

From the repository root in PowerShell:

```powershell
Copy-Item .env.example .env
# Edit .env and provide the real LLAMA_CLOUD_API_KEY value.
uv sync
uv run uvicorn document_rag.web.app:app --reload --env-file .env
```

Open <http://127.0.0.1:8000>. The first model downloads can take several minutes. A CUDA-capable
GPU is strongly recommended for Qwen generation; `device_map=auto` can fall back to available
hardware, but CPU generation will be slow.

## Real financial-PDF smoke test

1. Choose a financial report PDF whose text contains at least one easily verified value and page
   number, such as annual revenue or operating income. Record the expected value and its PDF page.
2. Start the application with the command above and open <http://127.0.0.1:8000>.
3. Upload the PDF. Confirm that the page reports the filename, page count, chunk count, and an
   **Ask this document** form. This also confirms that both in-memory indexes were built.
4. Ask a supported question whose answer is explicit in the PDF, for example: `What total revenue
   was reported for 2025?`
5. Confirm that the page shows:
   - a non-empty answer;
   - source citations containing both page numbers and chunk IDs;
   - no more than five unique retrieved chunks in the debugging section;
   - retrieved text that contains the evidence used for the answer.
6. Compare the answer and cited page with the value recorded in step 1. Treat a wrong calculation
   or unsupported claim as a failed smoke test even if retrieval found the correct page.
7. Ask a deliberately unsupported question, such as one about a company or reporting year absent
   from the PDF. Confirm that the answer says it cannot be supported by the supplied context rather
   than inventing a value.
8. Upload a different PDF and confirm that subsequent questions cite only chunks from the new
   document. This verifies session-index replacement rather than persistence.

For reproducibility, record the PDF filename and SHA-256, question text, answer, citations, model
IDs/revisions, device, and application commit SHA with the smoke-test result.

## Automated tests

Unit tests inject fake parser, retriever, embedder, and generator components. They do not download
models, contact LlamaParse or Hugging Face, or require a GPU.
