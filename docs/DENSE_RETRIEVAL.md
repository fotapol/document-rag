# Dense retrieval baseline

This experiment replaces BM25 scoring with exact dense cosine similarity while keeping the
benchmark corpus, questions, source-element lineage, gold evidence, evaluator, test split, and
K values unchanged.

## Configuration

- Embedding model: `BAAI/bge-small-en-v1.5`
- Pinned revision: `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`
- Embedding dimension: 384
- Document encoding: unmodified `DocumentChunk.text` through
  `SentenceTransformer.encode_document`
- Query encoding: question text only through `SentenceTransformer.encode_query`, prefixed with
  `Represent this sentence for searching relevant passages: `
- Storage: unquantized `float32` NumPy vectors
- Normalization: L2 unit normalization after encoding
- Similarity: exact matrix-vector dot product, equivalent to cosine similarity for unit vectors
- Retrieval scope: chunks belonging to the question's document
- Tie breaking: descending similarity, then ascending deterministic `chunk_id`

The query never includes the reference answer, reasoning program, or supporting evidence.

## Run the benchmark

From the repository root in PowerShell:

```powershell
uv run document-rag retrieval dense `
  --finqa data/processed/finqa `
  --docfinqa data/processed/docfinqa-v2 `
  --bm25-metrics artifacts/retrieval/bm25/retrieval_metrics.json `
  --output artifacts/retrieval/dense-bge-small-en-v1.5 `
  --embedding-model BAAI/bge-small-en-v1.5 `
  --model-revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a `
  --batch-size 32 `
  --device cpu
```

The command embeds 63,473 existing chunks and 2,038 existing test questions. The first run may
download the pinned model and create the local document-embedding cache. CPU execution is fully
supported; use `--device cuda` only when a CUDA-enabled PyTorch installation is available.

The recorded benchmark was run with `--batch-size 32 --device cuda` on an NVIDIA GeForce RTX
2060. The metrics artifact records the resolved device as `cuda:0`.

The output directory contains:

- `dense_predictions.jsonl`: per-query rankings and source lineage;
- `dense_metrics.json`: dense metrics and complete experiment metadata;
- `retrieval_comparison.json`: BM25, dense, and dense-minus-BM25 metrics for FinQA, DocFinQA,
  and the combined benchmark;
- `embedding_cache/embeddings.npy`: normalized document vectors;
- `embedding_cache/embedding_manifest.json`: cache identity and configuration.

Embedding cache directories are ignored by Git. A cache is reused only when its model identity,
model revision, corpus fingerprint, ordered chunk IDs, batch size, device, encoding strategy,
dimension, dtype, and normalization all match. A mismatch or corrupt cache causes re-encoding.

## Results

The values below use the frozen BM25 artifact and report absolute dense-minus-BM25 deltas.
Hit Rate and Recall values are percentages; their deltas are percentage points.

| Scope | Retriever | Hit@1 | Hit@3 | Hit@5 | Recall@1 | Recall@3 | Recall@5 | MRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FinQA | BM25 | 49.96 | 76.90 | 86.92 | 42.55 | 70.39 | 81.63 | 0.6435 |
| FinQA | Dense | 65.65 | 92.15 | 97.12 | 57.57 | 86.21 | 92.84 | 0.7893 |
| FinQA | Delta | +15.69 | +15.26 | +10.20 | +15.02 | +15.82 | +11.21 | +0.1458 |
| DocFinQA | BM25 | 43.10 | 64.42 | 71.27 | 37.17 | 57.15 | 64.37 | 0.5412 |
| DocFinQA | Dense | 22.00 | 37.49 | 45.90 | 18.31 | 31.42 | 38.99 | 0.3090 |
| DocFinQA | Delta | -21.10 | -26.94 | -25.36 | -18.86 | -25.74 | -25.37 | -0.2322 |
| Combined | BM25 | 46.96 | 71.44 | 80.08 | 40.20 | 64.60 | 74.08 | 0.5987 |
| Combined | Dense | 46.57 | 68.25 | 74.73 | 40.41 | 62.25 | 69.30 | 0.5793 |
| Combined | Delta | -0.39 | -3.19 | -5.35 | +0.21 | -2.35 | -4.78 | -0.0194 |

Dense retrieval clearly improves FinQA but does not outperform the lexical baseline on
DocFinQA. The combined result is nearly tied at rank one and worse at larger K values because
the DocFinQA regression outweighs the FinQA gain. Explaining that dataset gap requires a later
error analysis; changing chunking, prompts, models, or ranking in this controlled baseline would
confound the comparison.

## Optional real-model smoke test

This command intentionally requires Hugging Face access on the first run and is not part of the
offline unit test suite:

```powershell
uv run python -c "from document_rag.retrieval.embeddings import SentenceTransformerEmbedder; e = SentenceTransformerEmbedder(device='cpu'); print(e.model_id, e.model_revision, e.dimension, e.embed_documents(('Operating income increased.',), batch_size=1).shape, e.embed_queries(('What happened to operating income?',), batch_size=1).shape)"
```

## Scope

This is a single-model, exact dense baseline. It does not tune BM25, use approximate nearest
neighbors, combine lexical and semantic scores, rerank results, or fine-tune the embedding model.
