import json
from pathlib import Path

import pytest

from document_rag.datasets import (
    DatasetConfig,
    DatasetFiles,
    DatasetName,
    DatasetSplit,
)
from document_rag.datasets.finqa import prepare_finqa_dataset


def make_config() -> DatasetConfig:
    return DatasetConfig(
        name=DatasetName.FINQA,
        schema_version="1",
        source_url="https://github.com/example/finqa.git",
        source_revision="a" * 40,
        files=DatasetFiles(
            train="dataset/train.json",
            validation="dataset/dev.json",
            test="dataset/test.json",
        ),
    )


def make_raw_record() -> dict[str, object]:
    return {
        "pre_text": [
            "Revenue increased during the year.",
        ],
        "post_text": [
            "Additional information follows.",
        ],
        "filename": "ABC/2025/page_10.pdf",
        "table_ori": [
            ["", "2024", "2025"],
            ["Revenue", "$100", "$120"],
        ],
        "table": [
            ["", "2024", "2025"],
            ["revenue", "$ 100", "$ 120"],
        ],
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
            "gold_inds": {
                "table_1": "revenue ; $ 100 ; $ 120",
            },
            "exe_ans": 20,
            "program_re": "subtract(120, 100)",
        },
        "id": "ABC/2025/page_10.pdf-1",
    }


def write_train_source(source_directory: Path) -> None:
    dataset_directory = source_directory / "dataset"
    dataset_directory.mkdir(parents=True)

    (dataset_directory / "train.json").write_text(
        json.dumps([make_raw_record()]),
        encoding="utf-8",
    )


def test_service_prepares_selected_split_and_sample(
    tmp_path: Path,
) -> None:
    source_directory = tmp_path / "source"
    output_directory = tmp_path / "processed"

    write_train_source(source_directory)

    result = prepare_finqa_dataset(
        config=make_config(),
        source_directory=source_directory,
        output_directory=output_directory,
        splits=(DatasetSplit.TRAIN,),
    )

    assert len(result.splits) == 1
    assert result.splits[0].split is DatasetSplit.TRAIN

    assert result.splits[0].documents.record_count == 1
    assert result.splits[0].examples.record_count == 1

    assert result.manifest.path == (output_directory / "manifest.json")

    assert (output_directory / "train" / "documents.jsonl").is_file()
    assert (output_directory / "train" / "elements.jsonl").is_file()
    assert (output_directory / "train" / "examples.jsonl").is_file()
    assert (output_directory / "manifest.json").is_file()

    assert len(result.sample_splits) == 1
    assert result.sample_splits[0].split is DatasetSplit.TRAIN
    assert result.sample_splits[0].examples.record_count == 1

    assert result.sample_manifest is not None
    assert result.sample_manifest.path == (output_directory / "sample" / "manifest.json")

    assert (output_directory / "sample" / "train" / "documents.jsonl").is_file()
    assert (output_directory / "sample" / "train" / "elements.jsonl").is_file()
    assert (output_directory / "sample" / "train" / "examples.jsonl").is_file()
    assert (output_directory / "sample" / "manifest.json").is_file()

    assert result.report_overlaps == ()


def test_service_is_reproducible(tmp_path: Path) -> None:
    source_directory = tmp_path / "source"
    output_directory = tmp_path / "processed"

    write_train_source(source_directory)

    first = prepare_finqa_dataset(
        config=make_config(),
        source_directory=source_directory,
        output_directory=output_directory,
        splits=(DatasetSplit.TRAIN,),
    )

    second = prepare_finqa_dataset(
        config=make_config(),
        source_directory=source_directory,
        output_directory=output_directory,
        splits=(DatasetSplit.TRAIN,),
    )

    assert first.manifest.checksum_sha256 == second.manifest.checksum_sha256

    assert first.sample_manifest is not None
    assert second.sample_manifest is not None

    assert first.sample_manifest.checksum_sha256 == second.sample_manifest.checksum_sha256

    assert first.splits[0].documents.checksum_sha256 == second.splits[0].documents.checksum_sha256
    assert first.splits[0].elements.checksum_sha256 == second.splits[0].elements.checksum_sha256
    assert first.splits[0].examples.checksum_sha256 == second.splits[0].examples.checksum_sha256

    assert (
        first.sample_splits[0].documents.checksum_sha256
        == second.sample_splits[0].documents.checksum_sha256
    )
    assert (
        first.sample_splits[0].elements.checksum_sha256
        == second.sample_splits[0].elements.checksum_sha256
    )
    assert (
        first.sample_splits[0].examples.checksum_sha256
        == second.sample_splits[0].examples.checksum_sha256
    )


def test_service_rejects_empty_split_selection(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="At least one dataset split must be selected",
    ):
        prepare_finqa_dataset(
            config=make_config(),
            source_directory=tmp_path / "source",
            output_directory=tmp_path / "processed",
            splits=(),
        )


def test_service_rejects_duplicate_splits(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="Dataset splits must be unique",
    ):
        prepare_finqa_dataset(
            config=make_config(),
            source_directory=tmp_path / "source",
            output_directory=tmp_path / "processed",
            splits=(
                DatasetSplit.TRAIN,
                DatasetSplit.TRAIN,
            ),
        )
