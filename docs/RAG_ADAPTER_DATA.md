# RAG-aligned adapter-v2 data

The original training exporter intentionally reproduces the FinQA and DocFinQA benchmark shape:
gold facts are rendered as an unlabeled context and the assistant returns only the reference answer.
That format is useful for reproducing adapter v1, but it teaches outputs such as `14.1` rather than
the grounded application response `$14.1 million. [Source 2]`.

The `training export-rag` command is a separate schema-v2 pipeline for adapter v2. It leaves the
legacy exporter unchanged and reuses the production RAG prompt, table-row representation, hybrid
retrieval factory, source lineage, and diversity configuration.

## Data flow

```text
prepared FinQA + prepared DocFinQA
-> document-scoped BM25 + dense + RRF retrieval
-> parent-aware table-row diversity
-> production top-K context
-> gold-lineage sufficiency check
-> supported, oracle-augmented, or refusal example
-> conservative unit reconstruction
-> production system/user messages + assistant target
-> deterministic split JSONL + manifest
```

This pipeline prepares generator training data. It does not train an embedding model or modify the
frozen retrieval benchmarks.

## Supported examples

A supported target is emitted only when every gold source element is represented by a context
chunk. When production retrieval contains all evidence, `context_mode` is `retrieved`. Otherwise,
the default policy inserts the missing gold chunks, removes the lowest non-gold chunks needed to
stay inside `DOCUMENT_RAG_TOP_K`, and records `context_mode="oracle_augmented"`.

Oracle augmentation is explicit because generator supervision must never pair an answer with a
context that cannot support it. It is not evidence that production retrieval succeeded. Use the
`oracle_augmented_count` in the manifest as a retrieval-quality diagnostic.

The assistant target contains:

- the reference answer or available reference explanation;
- preserved or conservatively inferred units;
- every `[Source N]` label whose chunk carries gold source-element lineage.

Example:

```json
{
  "answerable": true,
  "category": "reasoning",
  "context_mode": "oracle_augmented",
  "expected_units": ["$", "million"],
  "gold_source_numbers": [1],
  "messages": [
    {"role": "system", "content": "<the exact production system instruction>"},
    {"role": "user", "content": "Retrieved context:\n\n[Source 1 | page 1 | chunk_id ...]\n...\n\nQuestion:\n..."},
    {"role": "assistant", "content": "The answer is $20 million. [Source 1]"}
  ]
}
```

`reasoning_program` is retained as audit metadata but is not inserted into the model prompt. The
model is supervised on a concise grounded answer, not a long hidden chain of thought.

## Unit policy

Units are part of the target contract rather than a post-processing decoration.

The exporter uses this conservative order:

1. Preserve currency symbols, percentages, scales, or basis points already present in the answer.
2. Infer units explicitly requested by the question, such as “in millions of dollars.”
3. Infer units attached to an exact occurrence of the answer in gold evidence.
4. For addition/subtraction programs, preserve a dollar dimension when the gold evidence is
   monetary.
5. Treat an answer as unitless when neither the question nor evidence indicates an applicable unit.
6. Exclude a supported example when evidence contains financial units but the correct relationship
   to a bare numeric answer is ambiguous.

An example whose gold evidence requires more independent chunks than `DOCUMENT_RAG_TOP_K` is also
excluded from supported supervision and counted under `gold_source_overflow_exclusion_count`.
Truncating those sources would teach an answer that the configured production prompt cannot fully
support.

For example, `380` cannot safely be converted from evidence containing `3.8%` without knowing
whether the intended answer is `380 basis points`, `3.8%`, or a dataset-specific normalization.
Such a record is counted under `ambiguous_unit_exclusion_count` instead of becoming incorrect
supervision. Review these exclusions to add dataset-specific rules only when the semantics are
verified.

## Refusal examples

For a supported question, the exporter also constructs a candidate context with all gold chunks
removed from the deeper retrieved pool. It emits a refusal only when:

- at least one non-gold context chunk remains;
- no gold lineage remains;
- the reference answer does not occur in the remaining context.

The target is exactly:

```text
I cannot answer this question from the supplied context.
```

Candidates are selected deterministically to approach `--refusal-ratio` (default `0.2`). The
actual count can be lower when safe negative contexts are unavailable. Review a sample manually
because absence of the reference string cannot prove that a semantically equivalent answer is
absent.

## Document isolation

The exporter derives a stable `document_group_id` from the source report. FinQA page paths such as
`ACME/2025/page_10.pdf` are grouped at `ACME/2025`, preventing different pages of one annual report
from crossing train, validation, and test. DocFinQA `source_file` metadata uses the same rule.

The default export does not trust the official question-level splits. It loads all source splits
and deterministically assigns every source-report group to an 80/10/10 train/validation/test split
using SHA-256. The original split remains available as `source_split` for auditing. Consequently,
all pages and linked FinQA/DocFinQA examples from the same report receive one output split even when
the upstream datasets placed them differently.

`--no-document-resplit` exists for focused diagnostics. In that mode, the export fails if a
source-report group appears in more than one selected input split. Do not use that mode for final
adapter data unless the inputs have already been grouped safely.

The real diagnostic PDF and the frozen RAG evaluation suite must remain outside training.

## Local export

Prepare both normalized datasets first. Then, from the repository root:

```powershell
uv run document-rag training export-rag `
  --finqa data/processed/finqa `
  --docfinqa data/processed/docfinqa `
  --output artifacts/training/rag-adapter-v2 `
  --refusal-ratio 0.2
```

The command exports all three deterministic report-level splits by default. It loads
the pinned BGE embedding model used by the application; it does not load Qwen or require a GPU.
The first embedding-model download requires Hugging Face access, while later runs can use the local
cache. `DOCUMENT_RAG_*` values from the environment control retrieval, including top-K, candidate
depth, model revision, and the table-parent cap.

For a quick development run, append `--split train`; the exporter still reads every upstream split
before selecting the assigned training records. To inspect behavior without oracle augmentation,
use `--no-oracle-augment`; questions whose retrieved context is insufficient will then contribute
only eligible refusal examples.

Outputs:

```text
artifacts/training/rag-adapter-v2/
  train.jsonl
  validation.jsonl
  test.jsonl
  manifest.json
```

The output is intentionally under the ignored `artifacts/` directory. Commit the exporter, tests,
and documentation—not generated training data.

## Audit before Kaggle

Before uploading anything, inspect `manifest.json` and a sample from every split. Check:

- supported/refusal ratio;
- retrieved versus oracle-augmented counts;
- ambiguous-unit exclusions;
- gold-source overflow exclusions;
- simple, table, and reasoning category balance;
- correct units and source labels in assistant targets;
- exact refusal targets;
- no diagnostic-suite document groups;
- pinned embedding model and revision;
- artifact SHA-256 values.

Do not proceed if most examples are oracle augmented: that indicates retrieval still does not
produce realistic contexts. Do not weaken unit validation merely to increase record count.

## Kaggle handoff

Kaggle should execute training, not own preprocessing logic.

1. Upload the four frozen output files as a private, versioned Kaggle Dataset.
2. Record the repository commit and manifest SHA-256 in the notebook configuration.
3. Load the JSONL `messages` field without reconstructing prompts in the notebook.
4. Apply the pinned Qwen chat template with `add_generation_prompt=False`.
5. Mask system and user tokens so loss is calculated only on the assistant response.
6. Train a new adapter ID or revision; never overwrite adapter v1.
7. Evaluate base, adapter v1, and adapter v2 against identical frozen RAG contexts.
8. Publish v2 only if grounded accuracy, units, citations, calculations, and refusals meet the
   predeclared acceptance thresholds.

Training is the next milestone after the generated artifacts have been audited and the frozen
evaluation suite has been expanded. The manifest makes the exact data used by Kaggle reproducible.
