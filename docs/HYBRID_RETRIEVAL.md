# Hybrid retrieval baseline

This controlled experiment combines the frozen BM25 and dense rankings with unweighted
Reciprocal Rank Fusion (RRF). It keeps the benchmark corpus, questions, per-document retrieval
scope, source-element lineage, gold evidence, evaluator, and K values unchanged.

## Fusion configuration

For each candidate chunk `d`, the fused score is:

```text
RRF(d) = sum(1 / (rrf_k + component_rank(d)))
```

The recorded baseline uses:

- `rrf_k = 60`;
- `candidate_k = 20` from BM25 and 20 from dense retrieval;
- evaluation K values `1, 3, 5`;
- unweighted rank contributions only—raw BM25 and cosine scores are never combined;
- deduplication by deterministic `chunk_id`;
- final ordering by descending RRF score, then ascending `chunk_id`;
- dense model `BAAI/bge-small-en-v1.5` at revision
  `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`.

A shared chunk receives one contribution from each component ranking. A chunk found by only one
retriever receives one contribution. Hybrid predictions record `bm25_rank`, `dense_rank`,
`rrf_score`, and the existing source-element lineage alongside the generic retrieval fields.

## Run the benchmark

From the repository root in PowerShell:

```powershell
uv run document-rag retrieval hybrid `
  --finqa data/processed/finqa `
  --docfinqa data/processed/docfinqa-v2 `
  --bm25-metrics artifacts/retrieval/bm25/retrieval_metrics.json `
  --dense-metrics artifacts/retrieval/dense-bge-small-en-v1.5/dense_metrics.json `
  --output artifacts/retrieval/hybrid-rrf-k60-c20 `
  --rrf-k 60 `
  --candidate-k 20 `
  --embedding-model BAAI/bge-small-en-v1.5 `
  --model-revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a `
  --batch-size 32 `
  --device cuda `
  --cache-directory artifacts/retrieval/dense-bge-small-en-v1.5/embedding_cache
```

CPU execution is supported; use a separate CPU cache path when the existing cache was produced
on CUDA. The recorded experiment used `cuda:0` on an NVIDIA GeForce RTX 2060 and reused the
existing 63,473-row dense cache. Cache validation remains controlled by the
dense model, revision, corpus identity, ordered chunk IDs, batch size, device, encoding strategy,
dimension, dtype, and normalization.

The output directory contains:

- `hybrid_predictions.jsonl`: per-query fused rankings with component-rank diagnostics;
- `hybrid_metrics.json`: hybrid metrics, corpus identity, and complete fusion configuration;
- `retrieval_comparison.json`: BM25, dense, hybrid, hybrid-minus-BM25, and
  hybrid-minus-dense metrics for every benchmark scope.

## Results

Hit Rate and Recall are percentages.

| Scope | Retriever | Hit@1 | Hit@3 | Hit@5 | Recall@1 | Recall@3 | Recall@5 | MRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FinQA | BM25 | 49.96 | 76.90 | 86.92 | 42.55 | 70.39 | 81.63 | 0.6435 |
| FinQA | Dense | 65.65 | 92.15 | 97.12 | 57.57 | 86.21 | 92.84 | 0.7893 |
| FinQA | Hybrid | 57.63 | 87.36 | 95.38 | 49.36 | 81.07 | 90.55 | 0.7286 |
| DocFinQA | BM25 | 43.10 | 64.42 | 71.27 | 37.17 | 57.15 | 64.37 | 0.5412 |
| DocFinQA | Dense | 22.00 | 37.49 | 45.90 | 18.31 | 31.42 | 38.99 | 0.3090 |
| DocFinQA | Hybrid | 31.31 | 54.77 | 62.74 | 26.37 | 47.44 | 55.25 | 0.4362 |
| Combined | BM25 | 46.96 | 71.44 | 80.08 | 40.20 | 64.60 | 74.08 | 0.5987 |
| Combined | Dense | 46.57 | 68.25 | 74.73 | 40.41 | 62.25 | 69.30 | 0.5793 |
| Combined | Hybrid | 46.12 | 73.11 | 81.11 | 39.31 | 66.36 | 75.12 | 0.6007 |

Absolute hybrid deltas are percentage points except MRR.

| Scope | Baseline | Hit@1 | Hit@3 | Hit@5 | Recall@1 | Recall@3 | Recall@5 | MRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FinQA | Hybrid - BM25 | +7.67 | +10.46 | +8.46 | +6.81 | +10.68 | +8.92 | +0.0851 |
| FinQA | Hybrid - Dense | -8.02 | -4.80 | -1.74 | -8.21 | -5.14 | -2.29 | -0.0608 |
| DocFinQA | Hybrid - BM25 | -11.78 | -9.65 | -8.53 | -10.79 | -9.72 | -9.12 | -0.1050 |
| DocFinQA | Hybrid - Dense | +9.32 | +17.28 | +16.84 | +8.06 | +16.02 | +16.26 | +0.1272 |
| Combined | Hybrid - BM25 | -0.83 | +1.67 | +1.03 | -0.89 | +1.76 | +1.04 | +0.0020 |
| Combined | Hybrid - Dense | -0.44 | +4.86 | +6.38 | -1.10 | +4.11 | +5.82 | +0.0214 |

## Interpretation

RRF moves each dataset toward the stronger component but does not match that component. FinQA
remains well above BM25 but below dense retrieval; DocFinQA recovers substantially from dense but
remains below BM25.

The combined benchmark improves over both baselines at ranks 3 and 5. In particular, hybrid
Recall@5 is 75.12%, compared with 74.08% for BM25 and 69.30% for dense retrieval. Combined MRR is
also narrowly highest at 0.6007. However, hybrid Hit@1 is 46.12%, below BM25's 46.96% and dense
retrieval's 46.57%, so this baseline does not improve the single-context retrieval case.

This result supports later investigation of fusion or routing, but no parameter sweep, weighted
fusion, dataset-specific configuration, reranking, or BM25/dense tuning belongs to this baseline.
