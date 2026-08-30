# Document RAG

Auditable retrieval-augmented generation system for financial and business documents.

## Status

Version `0.1.0` is under active development.

The current repository contains the typed Python project foundation, automated
quality checks, tests, and continuous integration.

## Planned capabilities

- Process narrative text and financial tables.
- Retrieve evidence using lexical, dense, and hybrid search.
- Fine-tune a financial evidence reranker.
- Generate grounded answers with page-level citations.
- Perform traceable numerical calculations.
- Expose the pipeline through FastAPI.

## Development requirements

- Git
- uv
- Python 3.12

`uv` installs and manages the required Python version and project dependencies.

## Setup

```shell
git clone https://github.com/fotapol/document-rag.git
cd document-rag
uv sync
```

## Quality checks

```shell
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

The same checks run automatically on every `push` and `pull_request` to `main`.

## Run the RAG MVP

Create `.env` from `.env.example`, set `LLAMA_CLOUD_API_KEY`, then start the
application from the repository root:

```powershell
Copy-Item .env.example .env
uv sync
uv run uvicorn document_rag.web.app:app --reload --env-file .env
```

Open `http://127.0.0.1:8000`, upload one financial PDF, and ask a question about
it. The detailed configuration and real smoke-test procedure are documented in
[`docs/RAG_MVP.md`](docs/RAG_MVP.md).

Uploaded tables are indexed as canonical row-level retrieval units so a query
cannot accidentally combine values from different rows. The row format, parent
lineage, deduplication, and small-corpus BM25 behavior are documented in
[`docs/TABLE_AWARE_RETRIEVAL.md`](docs/TABLE_AWARE_RETRIEVAL.md).

## Project Structure
```
document-rag/
├── src/document_rag/   # Application package
├── tests/              # Automated tests
├── .github/workflows/  # Continuous integration
├── pyproject.toml      # Project and tool configuration
└── uv.lock             # Reproducible dependency lock file
```
## License

This project is licensed under the MIT License.
