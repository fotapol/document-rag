"""Tests for normalized-data BM25 benchmark artifacts."""

import json
from pathlib import Path

from pydantic import BaseModel

from document_rag.cli import main
from document_rag.datasets.models import (
    DatasetExample,
    DatasetName,
    DatasetSplit,
    ReferenceAnswer,
    SupportingFact,
)
from document_rag.domain.documents import (
    Document,
    DocumentElement,
    DocumentElementType,
    TableCoordinates,
)
from document_rag.domain.questions import Question
from document_rag.retrieval.benchmark import run_bm25_benchmark


def write_jsonl(path: Path, records: tuple[BaseModel, ...]) -> None:
    """Write Pydantic fixtures using the normalized JSONL format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n" for record in records
        ),
        encoding="utf-8",
        newline="\n",
    )


def build_normalized_finqa(root: Path) -> str:
    """Create one benchmark-ready normalized FinQA split."""

    document_id = "finqa:report/page_1.pdf"
    table_id = f"{document_id}:table"
    row_id = f"{document_id}:table_1"
    document = Document(
        document_id=document_id,
        file_name="page_1.pdf",
        page_count=1,
        metadata={"dataset": DatasetName.FINQA.value},
    )
    paragraph = DocumentElement(
        element_id=f"{document_id}:text_0",
        document_id=document_id,
        element_type=DocumentElementType.PARAGRAPH,
        source_text="Cash and cash equivalents declined.",
        page_number=1,
    )
    punctuation = DocumentElement(
        element_id=f"{document_id}:text_1",
        document_id=document_id,
        element_type=DocumentElementType.PARAGRAPH,
        source_text=".",
        page_number=1,
    )
    table = DocumentElement(
        element_id=table_id,
        document_id=document_id,
        element_type=DocumentElementType.TABLE,
        source_text="Metric | 2024 | 2025\nOperating income | 40 | 55",
        page_number=1,
        table_coordinates=TableCoordinates(table_id=table_id),
    )
    row = DocumentElement(
        element_id=row_id,
        document_id=document_id,
        element_type=DocumentElementType.TABLE_ROW,
        source_text="Operating income | 40 | 55",
        page_number=1,
        parent_element_id=table_id,
        table_coordinates=TableCoordinates(table_id=table_id, row_index=1),
    )
    example = DatasetExample(
        dataset=DatasetName.FINQA,
        split=DatasetSplit.TEST,
        example_id="finqa:example-1",
        question=Question(
            question_id="finqa:example-1",
            document_id=document_id,
            text="What was operating income in 2025?",
        ),
        reference_answer=ReferenceAnswer(text="55"),
        supporting_facts=(SupportingFact(source_key="table_1", element_id=row_id),),
    )

    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({"dataset": DatasetName.FINQA.value}),
        encoding="utf-8",
    )
    split_directory = root / DatasetSplit.TEST.value
    write_jsonl(split_directory / "documents.jsonl", (document,))
    write_jsonl(
        split_directory / "elements.jsonl",
        (paragraph, punctuation, table, row),
    )
    write_jsonl(split_directory / "examples.jsonl", (example,))
    return row_id


def test_benchmark_writes_deterministic_lineage_based_artifacts(
    tmp_path: Path,
) -> None:
    """The benchmark should rank table evidence and serialize stable metrics."""

    dataset_root = tmp_path / "finqa"
    row_id = build_normalized_finqa(dataset_root)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    first = run_bm25_benchmark(
        dataset_directories={DatasetName.FINQA: dataset_root},
        output_directory=first_output,
    )
    second = run_bm25_benchmark(
        dataset_directories={DatasetName.FINQA: dataset_root},
        output_directory=second_output,
    )

    assert first.predictions_path.read_bytes() == second.predictions_path.read_bytes()
    assert first.metrics_path.read_bytes() == second.metrics_path.read_bytes()
    assert first.metrics.hit_rate_at_k == ((1, 1.0), (3, 1.0), (5, 1.0))
    assert first.metrics.recall_at_k == ((1, 1.0), (3, 1.0), (5, 1.0))
    assert first.metrics.mrr == 1.0
    assert first.indexed_chunk_count == 2

    prediction = json.loads(first.predictions_path.read_text(encoding="utf-8"))
    assert prediction["retrieved"][0]["source_element_ids"] == [
        "finqa:report/page_1.pdf:table",
        row_id,
    ]

    metrics = json.loads(first.metrics_path.read_text(encoding="utf-8"))
    assert metrics["benchmark_config"] == {
        "k_values": [1, 3, 5],
        "retrieval_scope": "question_document",
        "retriever_type": "bm25_okapi",
        "split": "test",
        "tokenization_strategy": "unicode_casefold_financial_lexical_v1",
    }
    assert metrics["dataset_counts"]["finqa"] == {
        "document_count": 1,
        "indexed_chunk_count": 2,
        "query_count": 1,
    }


def test_cli_command_runs_complete_benchmark(tmp_path: Path) -> None:
    """One project command should produce both required benchmark artifacts."""

    dataset_root = tmp_path / "finqa"
    output_root = tmp_path / "benchmark"
    build_normalized_finqa(dataset_root)

    exit_code = main(
        [
            "retrieval",
            "bm25",
            "--finqa",
            str(dataset_root),
            "--output",
            str(output_root),
        ]
    )

    assert exit_code == 0
    assert (output_root / "retrieval_predictions.jsonl").is_file()
    assert (output_root / "retrieval_metrics.json").is_file()
