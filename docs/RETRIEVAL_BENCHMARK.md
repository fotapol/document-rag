# BM25 retrieval benchmark

The first retrieval benchmark establishes a deterministic lexical baseline for
financial question answering. It answers whether BM25 can retrieve a chunk
that carries the question's gold source evidence within a selected rank cutoff,
without using an LLM, embeddings, a GPU, or an external service.

## Run the benchmark

Use prepared FinQA and/or DocFinQA directories. The command reads their
existing normalized artifacts; it does not download or normalize source data.

```shell
uv run document-rag retrieval bm25 \
  --finqa data/processed/finqa \
  --docfinqa data/processed/docfinqa \
  --output artifacts/retrieval/bm25 \
  --split test
```

The default cutoffs are 1, 3, and 5. To select different cutoffs, repeat
`--top-k`:

```shell
uv run document-rag retrieval bm25 \
  --finqa data/processed/finqa \
  --output artifacts/retrieval/bm25 \
  --top-k 1 \
  --top-k 5 \
  --top-k 10
```

The command creates:

- `retrieval_predictions.jsonl`, with one inspectable evaluation per question;
- `retrieval_metrics.json`, with aggregate and per-dataset metrics, benchmark
  configuration, and document, chunk, and query counts.

Both files use stable record ordering, sorted JSON keys, and UTF-8 encoding.

## Retrieval units

`BM25Retriever` indexes the existing `DocumentChunk` model and returns ranked
`RetrievalResult` records that reference the original chunk. Document identity,
page range, chunk text, and source-element IDs therefore remain available
without duplicating the chunk schema.

The CLI adapts normalized dataset elements into these retrieval units:

- each top-level paragraph or existing DocFinQA chunk becomes one
  `DocumentChunk`;
- a FinQA table becomes one chunk whose lineage contains the parent table ID
  and every child row ID represented in the table text;
- child rows are not indexed a second time as independent chunks.
- top-level elements with no lexical terms, such as punctuation-only FinQA
  placeholders, are omitted from the index.

This adapter preserves exact row-level gold evidence while avoiding a second
chunking pipeline. Indexes are scoped to `Question.document_id`, because the
normalized task already identifies the document being questioned. Counts in
the metrics artifact report the total chunks across these per-document indexes.

## Tokenization and ranking

The baseline uses `rank-bm25`'s BM25Okapi implementation. Its deterministic
lexical tokenizer case-folds Unicode text, keeps decimal-like values together,
and retains common currency and percent symbols as standalone terms. For
example, `Operating Income: $2.6M` becomes:

```text
operating, income, $, 2.6m
```

There is intentionally no stemming, lemmatization, stop-word model, or
language-specific preprocessing. Results sort by descending BM25 score, then
stable chunk ID for deterministic ties.

## Source-lineage relevance

A retrieved chunk is relevant only when its source-element IDs intersect the
gold supporting-element IDs:

```python
relevant = bool(
    set(result.chunk.source_element_ids)
    & set(gold_source_element_ids)
)
```

The benchmark does not infer relevance from text. FinQA row IDs are carried by
their parent table chunk, and DocFinQA supporting facts already reference its
normalized chunks, so no substring fallback is needed.

## Metrics

For each cutoff `K`:

- **Hit Rate@K** is the fraction of evaluated questions with at least one
  relevant chunk in the first `K` results.
- **Recall@K** is the macro-average fraction of each question's distinct gold
  source-element IDs covered by the first `K` chunks. Multiple chunks carrying
  the same evidence ID cannot inflate recall.
- **MRR** is the mean reciprocal rank of the first relevant result. A question
  with its first hit at rank 3 contributes `1/3`; a question with no hit
  contributes `0`.

Questions without gold source-element IDs are rejected as unevaluable instead
of silently lowering or inflating the reported metrics.

## Current limitations

This benchmark measures lexical retrieval within the document already linked
to each normalized question. It does not measure document routing across a
collection. Ranking quality depends on literal term overlap and deliberately
does not address synonyms or semantic similarity.

Dense retrieval, hybrid retrieval, vector databases, reranking, LLM inference,
answer generation, MLflow, and web UI changes are intentionally out of scope.
Later retrievers can reuse the evaluation functions by producing the same
`RetrievalResult` records.
