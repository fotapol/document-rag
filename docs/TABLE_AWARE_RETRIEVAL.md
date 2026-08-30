# Table-aware retrieval units

Financial table questions often contain several coordinates, such as a year, quarter, region,
and metric. Indexing a complete table fragment lets those terms match different rows. For example,
`2024 Q1` can occur in one row and `North America` in another, making the fragment look relevant
even though it does not contain the requested fact as one record.

The application index now expands HTML and Markdown tables into one `DocumentChunk` child per
data row. A row is rendered deterministically as:

```text
Table: Quarterly Revenue Detail
Quarter: 2024 Q1 | Region: North America | Revenue: $220,000
```

The table title and every column name are repeated in each row. This gives lexical and embedding
retrievers the schema needed to interpret otherwise ambiguous values. A repeated header is not a
data row, and exact duplicate rows produced by overlapping or split table fragments are indexed
only once.

## Identity and lineage

The original parsed chunk remains the parent record but is not indexed when it contains a table.
Each derived row has:

- a content-derived `chunk:table-row:...` identity;
- a content-derived row `source:table-row:...` source-element identity;
- the original page range;
- `retrieval_unit_kind="table_row"`;
- `parent_chunk_id` pointing to the original chunk;
- `parent_source_element_ids` containing the original parser/chunker lineage.

`TableRetrievalCorpus.parent_for()` and `ParentAwareRetriever.parent_for()` recover the complete
original chunk when later processing needs full-table context. Retrieval-result JSON includes the
parent fields for derived units. Ordinary chunks remain unchanged, so existing source-lineage
evaluation and frozen benchmark serialization retain their prior contracts.

If prose and a table share an original chunk, the prose becomes a separate `narrative` child and
stays searchable without carrying the table's multi-row contents. A heading by itself is not
treated as narrative content.

## Ranking behavior

The established offline BM25 benchmark remains frozen on its original Okapi configuration and
tokenizer. The uploaded-document application index explicitly uses:

- BM25+ so inverse-document-frequency weights stay positive in a small per-document corpus;
- deterministic query normalization that removes common question boilerplate;
- first-occurrence deduplication so a repeated term cannot be counted twice;
- unchanged financial tokens, including values, currencies, percentages, years, and quarters.

BM25+ is scoped to the application index because classic Okapi BM25 can assign zero or negative
weight to terms that occur in at least half of a tiny corpus. In that situation, matching more
query coordinates can lower a row's score. Dense retrieval still receives the original question,
and RRF still fuses the two rankings without combining their raw scores.

## Offline regression coverage

The tests use synthetic HTML and Markdown plus a deterministic fake embedder. They require no
Internet access, LlamaParse account, Hugging Face access, or GPU. The diagnostic table verifies:

- the `2024 Q1 / North America / $220,000` row ranks first in BM25+, dense, and RRF;
- terms from separate rows never appear in one indexed retrieval unit;
- row ranking and identity are deterministic;
- results contain no duplicate rows;
- page, row source identity, parent chunk identity, and parent source lineage survive retrieval;
- prose from a table-bearing chunk remains available for narrative questions.

Run the focused checks with:

```powershell
uv run pytest tests/unit/retrieval/test_table_units.py tests/unit/retrieval/test_bm25.py `
  tests/unit/rag/test_service.py --no-cov
```
