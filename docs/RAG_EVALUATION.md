# Frozen-context base versus LoRA evaluation

This evaluation answers one question: when retrieval and prompting are held constant, does the
financial LoRA improve or degrade grounded answer generation relative to its pinned Qwen base?

Every case stores its complete ranked retrieval results. The evaluator builds one grounded prompt,
records its SHA-256, and sends that same prompt through one loaded PEFT model twice:

1. with the adapter temporarily disabled for the base result;
2. with the adapter enabled for the LoRA result.

Both conditions therefore use the same tokenizer, base revision, context, prompt, device placement,
token limits, and greedy decoding (`do_sample=False`). The adapter is not merged into the base.

## Suite format

Suites are versioned JSON files. Each answerable case must specify at least one expected numeric
value or phrase and at least one required source number. Unsupported cases have no supported-answer
fields and must produce the exact grounded refusal.

```json
{
  "schema_version": 1,
  "name": "financial-report-diagnostic-v1",
  "document_sha256": "<PDF SHA-256>",
  "retrieval_config": {
    "type": "hybrid_rrf",
    "top_k": 5,
    "candidate_k": 20,
    "rrf_k": 60,
    "embedding_model_id": "BAAI/bge-small-en-v1.5",
    "embedding_model_revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
  },
  "cases": [
    {
      "case_id": "table-2024-q1-north-america",
      "category": "table_lookup",
      "context_mode": "production",
      "production_gold_rank": 1,
      "question": "What was North America revenue in 2024 Q1?",
      "expectation": {
        "answerable": true,
        "expected_values": ["220000"],
        "expected_units": ["$"],
        "required_source_numbers": [1]
      },
      "retrieved": [
        {
          "rank": 1,
          "score": 0.032,
          "chunk": {
            "chunk_id": "chunk:table-row:<stable-id>",
            "document_id": "sha256:<PDF SHA-256>",
            "document_sha256": "<PDF SHA-256>",
            "filename": "financial_report.pdf",
            "chunk_index": 3,
            "page_start": 3,
            "page_end": 3,
            "source_element_ids": ["source:table-row:<stable-id>"],
            "text": "Table: Quarterly Revenue Detail\nQuarter: 2024 Q1 | Region: North America | Revenue: $220,000",
            "char_count": 113,
            "token_count": 24,
            "block_count": 1,
            "retrieval_unit_kind": "table_row",
            "parent_chunk_id": "chunk:<parent-id>",
            "parent_source_element_ids": ["element:<source-id>"]
          }
        }
      ]
    }
  ]
}
```

Valid categories are `simple_lookup`, `table_lookup`, `reasoning`, and `unsupported`. Retrieval
ranks must be unique and contiguous from one, and a case cannot contain duplicate chunk IDs.

`context_mode="production"` means the context is exactly the application's top-K result.
`context_mode="oracle_augmented"` means verified gold evidence was inserted because production
retrieval missed it; `production_gold_rank` records its original rank when known. Oracle-augmented
cases isolate generator quality and must not be presented as end-to-end production accuracy.

## Metrics

The evaluator does not use the old “first number” heuristic. Every expected numeric value may
appear anywhere in an answer, so a correct explanation containing operands and a final result can
pass. Decimal formatting and thousands separators are normalized.

Each model receives separate metrics for:

- `content`: expected numbers and phrases, or the exact unsupported refusal;
- `numeric`: all expected numeric values occur in the answer;
- `units`: all required currency or scale markers occur;
- `citations`: required `[Source N]` labels occur and every cited number exists in the context;
- `refusal`: supported answers are not refused and unsupported answers use the exact refusal;
- `overall`: every applicable check passes.

Each metric records `correct`, `count`, and `accuracy`. `rag_metrics.json` also reports adapter-minus-
base deltas and metrics per question category and context mode. A scalar adapter answer can therefore pass numeric
correctness while failing units, citations, and overall grounded quality.

## Run command

From the repository root in PowerShell:

```powershell
uv run --no-sync document-rag rag evaluate `
  --suite artifacts/rag-evaluation/financial_report_suite.json `
  --output artifacts/rag-evaluation/results `
  --device-map auto
```

The command uses the pinned model IDs and revisions from `RAGConfig`. Optional CLI overrides are
available for the base model, base revision, adapter model, adapter revision, device map, maximum
input tokens, and maximum new tokens. Every resolved value is written to `rag_metrics.json`.

`--no-sync` is recommended on the current Windows CUDA setup because a normal `uv sync` may replace
the manually selected CUDA Torch wheel with the CPU build from the lock file.

Outputs:

- `rag_predictions.jsonl`: prompt, prompt hash, frozen context, both answers, and all per-case checks;
- `rag_metrics.json`: pinned configuration, suite identity, aggregate metrics, category metrics, and
  adapter-minus-base deltas.

The `artifacts/` directory is intentionally ignored by Git.

## Real financial-PDF procedure

Use the same PDF and frozen retrieved contexts for every comparison. For the current diagnostic PDF,
include at least these four cases:

1. Simple lookup: `What was the company's revenue in 2025?` Expected `$14.1 million`.
2. Large table: `What was North America revenue in 2024 Q1?` Expected `$220,000`.
3. Reasoning: `By how much did revenue increase from 2024 to 2025, in millions of dollars?`
   Expected `$1.7 million`.
4. Unsupported: `Who is the CEO of the company?` Expected the exact unsupported response.

First verify the expected facts directly in the PDF. Retrieve and save top-five row-aware RRF results
once; do not retrieve separately for base and adapter. Set each `required_source_numbers` entry to
the frozen source that actually contains the gold evidence. Run the command above and inspect both
artifact files rather than relying on aggregate accuracy alone.

Unit tests inject a fake paired generator and use synthetic contexts. They require no Internet,
LlamaParse account, Hugging Face access, GPU, Torch model load, or PEFT model load.

## Current diagnostic observation

A local CUDA run used PDF SHA-256
`4c75274c466fa8a82eaaaaf5815aff63184b751058b7deafad9ca7e17624254b` and suite SHA-256
`8130d2325f25ccf72d20612708846a2b9fac41ec8e7f9a485f371b1806e51ba2`. The saved source
artifact contained eight of the ten original parent chunks, so this is a diagnostic subset rather
than a publishable benchmark.

- Table lookup used production context; the `$220,000` row ranked first.
- Unsupported testing used production context.
- Simple lookup used oracle augmentation because annual revenue ranked 18th.
- Reasoning used oracle augmentation because the annual-comparison evidence ranked 17th.
- The pinned base passed all four overall checks.
- The pinned adapter passed all three numeric checks but passed zero unit checks, zero citation
  checks, and zero overall cases. It answered the unsupported CEO question with `John Doe`.

The result supports two separate follow-ups: add retrieval diversity or parent collapse so many rows
from one table cannot crowd out narrative evidence, and train a RAG-aligned adapter that preserves
units/citations and learns explicit unsupported examples. It does not justify retraining until a
larger frozen development suite is available.
