# Contributing

Thank you for considering an improvement to Document RAG. Bug reports, documentation fixes, tests,
and focused implementation changes are welcome.

## Before opening a change

- Search existing issues to avoid duplicate work.
- Open an issue before a large architectural change.
- Never include private PDFs, API keys, model weights, generated datasets, or benchmark artifacts.
- Keep the local single-user security boundary explicit in user-facing documentation.

## Development setup

```powershell
git clone https://github.com/fotapol/document-rag.git
Set-Location document-rag
uv sync --locked
uv run --no-sync pre-commit install
uv run --no-sync pre-commit install --hook-type pre-push
```

Copy `.env.example` to `.env` only for manual LlamaParse testing. Automated tests do not require
real credentials or external services.

## Making a pull request

1. Create a focused branch from the latest `main`.
2. Add or update offline tests for behavior changes.
3. Keep generated files under ignored `data/` or `artifacts/` directories.
4. Run the complete quality suite:

   ```powershell
   uv run --no-sync ruff check .
   uv run --no-sync ruff format --check .
   uv run --no-sync mypy
   uv run --no-sync pytest
   ```

5. Explain the motivation, implementation, verification, and user-visible limitations in the pull
   request.

The project uses typed Python 3.12, Ruff formatting, strict mypy checks, deterministic tests, and
small commits with imperative messages such as `feat: add ...` or `fix: preserve ...`.

## External-service and model changes

Changes involving LlamaParse, Hugging Face models, or CUDA must retain fake offline test boundaries.
Pin model revisions used in reproducible paths, document hardware assumptions, and do not make
network or GPU access a unit-test requirement.

## Reporting security problems

Do not disclose credentials, private documents, or exploitable vulnerabilities in a public issue.
Follow [SECURITY.md](SECURITY.md) instead.
