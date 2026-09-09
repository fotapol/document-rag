# Evaluation

The project separates retrieval evaluation from generator evaluation. Retrieval benchmarks ask
whether the correct source lineage appears in ranked chunks. Frozen-context generation evaluation
asks whether the LoRA improves grounded answers when retrieval and prompting are identical.

Generated predictions, embeddings, and metrics belong under the ignored `artifacts/` directory.

## Retrieval benchmark

The frozen test corpus contains:

| Dataset | Queries | Chunks |
|---|---:|---:|
| FinQA | 1,147 | 8,807 |
| DocFinQA | 891 | 54,666 |
| Combined | 2,038 | 63,473 |

A chunk is relevant only when its `source_element_ids` intersect the question's gold source IDs.
No answer-text or substring fallback is used.

Reported metrics are:

- **Hit Rate@K**: questions with at least one relevant chunk in the first K results;
- **Recall@K**: macro-average fraction of distinct gold source IDs covered by the first K; and
- **MRR**: mean reciprocal rank of the first relevant result.

Stable chunk IDs break equal-score ties, output records have deterministic ordering, and repeated
evidence IDs cannot inflate recall.

### Recorded results

Hit Rate and Recall are percentages.

| Scope | Retriever | Hit@1 | Hit@3 | Hit@5 | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FinQA | BM25 | 49.96 | 76.90 | 86.92 | 42.55 | 70.39 | 81.63 | 0.6435 |
| FinQA | Dense | 65.65 | 92.15 | 97.12 | 57.57 | 86.21 | 92.84 | 0.7893 |
| FinQA | Hybrid | 57.63 | 87.36 | 95.38 | 49.36 | 81.07 | 90.55 | 0.7286 |
| DocFinQA | BM25 | 43.10 | 64.42 | 71.27 | 37.17 | 57.15 | 64.37 | 0.5412 |
| DocFinQA | Dense | 22.00 | 37.49 | 45.90 | 18.31 | 31.42 | 38.99 | 0.3090 |
| DocFinQA | Hybrid | 31.31 | 54.77 | 62.74 | 26.37 | 47.44 | 55.25 | 0.4362 |
| Combined | BM25 | 46.96 | 71.44 | 80.08 | 40.20 | 64.60 | 74.08 | 0.5987 |
| Combined | Dense | 46.57 | 68.25 | 74.73 | 40.41 | 62.25 | 69.30 | 0.5793 |
| Combined | Hybrid | 46.12 | 73.11 | 81.11 | 39.31 | 66.36 | 75.12 | 0.6007 |

Dense retrieval substantially improves FinQA and regresses on DocFinQA. Hybrid RRF has the highest
combined Recall@5 and narrowly highest combined MRR, but it is not the strongest retriever for
every dataset or cutoff. The web application uses hybrid retrieval as a balanced baseline, not
because it dominates all component results.

### Commands

```powershell
uv run --no-sync document-rag retrieval bm25 `
  --finqa data/processed/finqa `
  --docfinqa data/processed/docfinqa `
  --output artifacts/retrieval/bm25 `
  --split test

uv run --no-sync document-rag retrieval dense `
  --finqa data/processed/finqa `
  --docfinqa data/processed/docfinqa `
  --bm25-metrics artifacts/retrieval/bm25/retrieval_metrics.json `
  --output artifacts/retrieval/dense `
  --embedding-model BAAI/bge-small-en-v1.5 `
  --model-revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a `
  --batch-size 32 `
  --device cpu

uv run --no-sync document-rag retrieval hybrid `
  --finqa data/processed/finqa `
  --docfinqa data/processed/docfinqa `
  --bm25-metrics artifacts/retrieval/bm25/retrieval_metrics.json `
  --dense-metrics artifacts/retrieval/dense/dense_metrics.json `
  --output artifacts/retrieval/hybrid `
  --rrf-k 60 `
  --candidate-k 20 `
  --embedding-model BAAI/bge-small-en-v1.5 `
  --model-revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a `
  --batch-size 32 `
  --device cpu
```

The first dense run downloads the pinned embedding model and creates a validated local embedding
cache. Cache identity includes the model revision, corpus fingerprint, chunk ordering, encoding
strategy, batch size, device, dimension, dtype, and normalization.

## Base-versus-adapter generation evaluation

Every frozen case stores its ranked retrieval context. The evaluator constructs one prompt and
runs the same loaded PEFT model twice:

1. with the adapter disabled for the base result; and
2. with the adapter enabled for the LoRA result.

The tokenizer, base revision, context, prompt, device, token limits, and deterministic decoding are
therefore identical. The adapter is never merged.

Answerable cases declare expected values or phrases, units, and required source numbers.
Unsupported cases require the exact response:

```text
I cannot answer this question from the supplied context.
```

The scorer separately checks value, unit, citation, calculation format, refusal behavior, and an
overall conjunction. Oracle-augmented context is labeled explicitly and measures generator
behavior, not end-to-end retrieval quality.

Run a versioned frozen suite with:

```powershell
uv run --no-sync document-rag rag evaluate `
  --suite artifacts/rag-evaluation/financial_report_suite.json `
  --output artifacts/rag-evaluation/results `
  --device-map auto
```

## Financial LoRA v4 validation

The published
[`fotapol/qwen3-1.7b-financial-rag-lora-v4`](https://huggingface.co/fotapol/qwen3-1.7b-financial-rag-lora-v4)
was compared with the pinned Qwen base on the same 900 schema-v6 validation prompts using greedy
decoding.

| Metric | Base | v4 | v4 - base |
|---|---:|---:|---:|
| Overall accuracy | 38.44% | 55.22% | +16.78 pp |
| Value accuracy | 41.25% | 55.42% | +14.17 pp |
| Unit accuracy | 41.94% | 91.00% | +49.07 pp |
| Citation completion | 33.33% | 83.47% | +50.14 pp |
| Calculation format | 0.00% | 90.83% | +90.83 pp |
| False-refusal rate | 41.39% | 4.44% | -36.94 pp |
| Unsupported refusal accuracy | 86.67% | 65.56% | -21.11 pp |

The adapter improves the intended RAG response format and overall validation score but is not
universally better. Refusal accuracy regresses relative to the base model, and financial-reasoning
overall accuracy remains low. These are release limitations, not hidden benchmark exceptions.
