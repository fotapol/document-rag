# Public release checklist

Use this checklist before changing repository visibility or creating a release tag.

## Repository safety

- [ ] Confirm `git status --short` is empty on the release commit.
- [ ] Run Gitleaks against the complete Git history with redaction enabled.
- [ ] Confirm `.env`, `data/`, `artifacts/`, PDFs, archives, and model weights are not tracked.
- [ ] Rotate any credential that was ever committed, logged, or shared publicly.
- [ ] Review repository members, deploy keys, webhooks, Actions secrets, and installed applications.

## Documentation and provenance

- [ ] Follow the README from a clean clone.
- [ ] Confirm the privacy and localhost-only deployment warnings remain accurate.
- [ ] Confirm all environment variables in `.env.example` are documented and contain no secret.
- [ ] Verify the live Hugging Face adapter card, license, revision, hashes, metrics, and limitations.
- [ ] Verify upstream model and dataset links, licenses, and citation guidance.
- [ ] Remove local paths, private hostnames, usernames, and unpublished benchmark claims.

## Build and verification

```powershell
uv lock --check
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy
uv run --no-sync pytest
uv build
```

- [ ] Inspect the source archive and wheel; the wheel must contain the web template.
- [ ] Install the wheel in a new temporary environment.
- [ ] Run `document-rag --help` and import the FastAPI app from that clean environment.
- [ ] Complete the real financial-PDF smoke test on supported hardware.
- [ ] Confirm CI passes both **Quality checks** and **Secret scan**.

## GitHub settings

- [ ] Change repository visibility only after the safety scan passes.
- [ ] Enable secret scanning and push protection.
- [ ] Enable Dependabot alerts and security updates.
- [ ] Enable private vulnerability reporting for the workflow documented in `SECURITY.md`.
- [ ] Add a repository description, website/model link, and topics such as `rag`,
  `financial-question-answering`, `fastapi`, `qwen`, and `lora`.
- [ ] Automatically delete head branches after pull requests are merged.

Protect `main` with a branch rule or ruleset:

- [ ] Require a pull request before merging.
- [ ] Require **Quality checks** and **Secret scan**.
- [ ] Require branches to be up to date before merging.
- [ ] Require conversation resolution.
- [ ] Block force pushes and branch deletion.
- [ ] Keep required approvals at zero unless another reviewer is available; a one-person project
  cannot approve its own pull request.
- [ ] Keep an administrator bypass available for recovery, but use it only deliberately.

## Release

- [ ] Merge the release-preparation pull request.
- [ ] Pull the protected `main` branch and repeat the build and smoke checks.
- [ ] Create annotated tag `v0.1.0`.
- [ ] Create a GitHub release summarizing features, hardware requirements, privacy boundaries,
  model limitations, and known issues.
- [ ] Verify all public links from an unauthenticated browser session.
