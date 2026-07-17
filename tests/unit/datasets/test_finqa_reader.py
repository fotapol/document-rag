import json
from pathlib import Path

import pytest

from document_rag.datasets import (
    DatasetConfig,
    DatasetFiles,
    DatasetName,
    DatasetSplit,
)
from document_rag.datasets.finqa import FinQARawReader


def make_config() -> DatasetConfig:
    return DatasetConfig(
        name=DatasetName.FINQA,
        schema_version="1",
        source_url="https://example.com/finqa.git",
        source_revision="a" * 40,
        files=DatasetFiles(
            train="dataset/train.json",
            validation="dataset/dev.json",
            test="dataset/test.json",
        ),
    )


def make_raw_record() -> dict[str, object]:
    return {
        "pre_text": ["Revenue increased."],
        "post_text": ["See the following table."],
        "filename": "ABC/2025/page_10.pdf",
        "table_ori": [["", "2024", "2025"], ["Revenue", "100", "120"]],
        "table": [["", "2024", "2025"], ["revenue", "100", "120"]],
        "qa": {
            "question": "How much did revenue increase?",
            "answer": "20",
            "explanation": "",
            "steps": [
                {
                    "op": "subtract",
                    "arg1": "120",
                    "arg2": "100",
                    "res": "20",
                }
            ],
            "program": "subtract(120, 100)",
            "gold_inds": {"table_1": "revenue ; 100 ; 120"},
            "exe_ans": 20,
            "program_re": "subtract(120, 100)",
        },
        "id": "ABC/2025/page_10.pdf-1",
    }


def test_reader_loads_json_records(tmp_path: Path) -> None:
    dataset_directory = tmp_path / "dataset"
    dataset_directory.mkdir()

    records = [make_raw_record()]
    (dataset_directory / "train.json").write_text(
        json.dumps(records),
        encoding="utf-8",
    )

    reader = FinQARawReader(
        source_directory=tmp_path,
        config=make_config(),
    )

    result = reader.read_split(DatasetSplit.TRAIN)

    assert result[0].id == "ABC/2025/page_10.pdf-1"
    assert result[0].qa.question == "How much did revenue increase?"


def test_reader_rejects_missing_file(tmp_path: Path) -> None:
    reader = FinQARawReader(
        source_directory=tmp_path,
        config=make_config(),
    )

    with pytest.raises(FileNotFoundError):
        reader.read_split(DatasetSplit.TRAIN)


def test_reader_rejects_non_array_json(tmp_path: Path) -> None:
    dataset_directory = tmp_path / "dataset"
    dataset_directory.mkdir()

    (dataset_directory / "train.json").write_text(
        json.dumps({"id": "example-1"}),
        encoding="utf-8",
    )

    reader = FinQARawReader(
        source_directory=tmp_path,
        config=make_config(),
    )

    with pytest.raises(ValueError, match="Expected a JSON array"):
        reader.read_split(DatasetSplit.TRAIN)


def test_reader_rejects_invalid_record_schema(tmp_path: Path) -> None:
    dataset_directory = tmp_path / "dataset"
    dataset_directory.mkdir()

    (dataset_directory / "train.json").write_text(
        json.dumps([{"id": "incomplete-record"}]),
        encoding="utf-8",
    )

    reader = FinQARawReader(
        source_directory=tmp_path,
        config=make_config(),
    )

    with pytest.raises(ValueError, match="Invalid FinQA data"):
        reader.read_split(DatasetSplit.TRAIN)
