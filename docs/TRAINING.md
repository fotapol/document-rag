# Adapter training

The repository prepares production-shaped supervision for financial RAG generation. It does not
train embeddings, alter frozen retrieval benchmarks, or require Qwen weights during export. GPU
training is intentionally a separate Kaggle step.

The published v4 adapter and its exact metadata are documented on
[Hugging Face](https://huggingface.co/fotapol/qwen3-1.7b-financial-rag-lora-v4).

## Why RAG-shaped supervision

The original FinQA and DocFinQA benchmark format pairs gold facts with short scalar answers such as
`14.1`. A deployed RAG answer instead needs a unit and citation, for example:

```text
The answer is $14.1 million. [Source 2]
```

`document-rag training export-rag` creates messages with the same system instruction, source
labels, table-row representation, retrieval stack, and context limits used by the application.

## Export flow

```text
prepared FinQA + DocFinQA
-> document-scoped BM25 + dense + RRF
-> parent-aware row diversity
-> gold-lineage sufficiency check
-> retrieved, oracle-augmented, or refusal context
-> conservative unit reconstruction
-> visible calculation supervision when available
-> production system/user messages and assistant target
-> exact Qwen chat-template token count
-> deterministic report-grouped splits and manifest
```

### Supported examples

A supported target is emitted only when every gold source element is represented in the context.
When production retrieval is incomplete, the default policy deterministically inserts missing gold
chunks and records `context_mode="oracle_augmented"`. Oracle augmentation prevents training on an
unsupported target; it is not evidence that production retrieval succeeded.

Injected evidence uses a deterministic source position rather than always becoming the final source.
Every example and manifest record exposes the injected chunk and source IDs for auditing.

### Units and calculations

Units are part of the target contract. The exporter:

1. preserves units already present in the answer;
2. uses units explicitly requested by the question;
3. inspects exact answer occurrences in gold evidence;
4. preserves monetary dimensions for compatible addition and subtraction; and
5. excludes examples whose unit relationship remains ambiguous.

When the dataset provides a reasoning program, it is retained as audit metadata and rendered as a
short visible calculation before the final answer. It is dataset-provided supervision, not hidden
chain of thought.

### Refusals

The exporter removes all gold chunks from a deeper candidate pool and creates a refusal example
only when non-gold context remains and the reference answer does not occur in it. The target is:

```text
I cannot answer this question from the supplied context.
```

Selection is deterministic and approaches the configured refusal ratio without inventing unsafe
negative examples.

### Sequence limits

The complete system, user, and assistant sequence is counted with the pinned Qwen tokenizer and chat
template. If it exceeds `--max-sequence-tokens`, the lowest-ranked non-gold chunks are removed and
source labels are rebuilt. Gold evidence is never silently removed. Examples that still cannot fit
are excluded and counted in the manifest.

### Document isolation

A stable report-level group ID prevents pages from the same annual report from crossing training,
validation, and test. The upstream question-level split remains recorded for audit, but final
assignment is deterministic at the source-report level.

## Prepare normalized datasets

The dataset commands download and normalize source records into the ignored `data/` directory.
Review command-specific options before a large run:

```powershell
uv run --no-sync document-rag data --help
uv run --no-sync document-rag data prepare --help
```

FinQA and DocFinQA retain their upstream provenance and source lineage. Generated data must not be
committed to this repository.

## Export training artifacts

```powershell
uv run --no-sync document-rag training export-rag `
  --finqa data/processed/finqa `
  --docfinqa data/processed/docfinqa `
  --output artifacts/training/rag-adapter `
  --refusal-ratio 0.2 `
  --max-sequence-tokens 4096
```

The current repository exporter writes schema version 5:

```text
artifacts/training/rag-adapter/
  train.jsonl
  validation.jsonl
  test.jsonl
  manifest.json
```

It loads the pinned BGE embedding model and Qwen tokenizer but not the Qwen language-model weights.
The first run needs Hugging Face access; later runs can use the local cache.

## Audit before training

Do not upload the output to Kaggle until the manifest and samples from every split have been
reviewed. Check:

- supported and refusal counts;
- retrieved versus oracle-augmented contexts;
- oracle source-position distribution;
- ambiguous-unit and gold-source-overflow exclusions;
- context trimming and sequence-overflow exclusions;
- simple lookup, table lookup, reasoning, and refusal balance;
- answer units and complete source labels;
- exact refusal targets;
- absence of the real diagnostic PDF;
- report-group isolation;
- pinned model revisions; and
- artifact SHA-256 values.

A high oracle-augmentation rate is a retrieval diagnostic. Do not describe it as production
retrieval success or weaken the unit rules merely to keep more records.

## Kaggle handoff

Kaggle supplies GPU compute; preprocessing logic remains versioned here.

1. Upload the frozen JSONL files and manifest as a private, versioned Kaggle Dataset.
2. Record the repository commit and manifest SHA-256 in the notebook configuration.
3. Load the existing `messages` field without rebuilding prompts.
4. Apply the pinned Qwen chat template with thinking disabled and no generation prompt.
5. Mask system and user tokens so loss is calculated only on the assistant target.
6. Assert that every sequence fits the manifest limit; do not silently filter examples.
7. Train a new adapter ID or revision rather than overwriting an earlier release.
8. Compare base and adapter on identical frozen validation and real-PDF contexts.
9. Publish only after reviewing value, unit, citation, calculation, and refusal metrics.

## Published v4 provenance

The published v4 adapter was trained for one epoch over 9,000 separately curated schema-v6
RAG-shaped examples:

- 40% direct lookup, including 3,287 table examples;
- 40% financial reasoning; and
- 20% unsupported-context refusal.

Training completed 1,125 optimizer steps using 4-bit NF4 QLoRA on two NVIDIA T4 GPUs with LoRA rank
16, alpha 32, dropout 0.05, effective batch size 8, and learning rate `1e-4`.

The v4 model card records hashes for the weights, adapter configuration, and training manifest. The
schema-v6 curation step is not produced by the repository's current schema-v5 exporter; this
distinction is stated explicitly so the public repository does not claim byte-for-byte
reproducibility that it does not provide.

See [Evaluation](EVALUATION.md) for the recorded base-versus-v4 results and limitations.
