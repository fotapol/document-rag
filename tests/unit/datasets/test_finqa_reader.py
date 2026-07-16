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


def test_reader_loads_json_records(tmp_path: Path) -> None:
    dataset_directory = tmp_path / "dataset"
    dataset_directory.mkdir()

    records = [{"id": "example-1"}, {"id": "example-2"}]
    (dataset_directory / "train.json").write_text(
        json.dumps(records),
        encoding="utf-8",
    )

    reader = FinQARawReader(
        source_directory=tmp_path,
        config=make_config(),
    )

    result = reader.read_split(DatasetSplit.TRAIN)

    assert result == records


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
