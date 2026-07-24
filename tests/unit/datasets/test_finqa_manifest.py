import json
from pathlib import Path

from document_rag.datasets import (
    DatasetConfig,
    DatasetFiles,
    DatasetName,
    DatasetSplit,
)
from document_rag.datasets.finqa import (
    PreparedFinQASplit,
    write_finqa_manifest,
    write_finqa_split,
)
from document_rag.domain import Document


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


def make_prepared_split() -> PreparedFinQASplit:
    document = Document(
        document_id="finqa:report.pdf",
        file_name="report.pdf",
        page_count=1,
    )

    return PreparedFinQASplit(
        split=DatasetSplit.TRAIN,
        documents=(document,),
        elements=(),
        examples=(),
    )


def test_manifest_records_source_and_artifacts(tmp_path: Path) -> None:
    written_split = write_finqa_split(
        make_prepared_split(),
        output_directory=tmp_path,
    )

    result = write_finqa_manifest(
        config=make_config(),
        written_splits=(written_split,),
        output_directory=tmp_path,
    )

    payload = json.loads(result.path.read_text(encoding="utf-8"))

    assert payload["dataset"] == "finqa"
    assert payload["source_revision"] == "a" * 40
    assert payload["splits"][0]["split"] == "train"
    assert payload["splits"][0]["documents"]["record_count"] == 1
    assert payload["splits"][0]["documents"]["path"] == "train/documents.jsonl"


def test_manifest_is_deterministic(tmp_path: Path) -> None:
    written_split = write_finqa_split(
        make_prepared_split(),
        output_directory=tmp_path,
    )

    first = write_finqa_manifest(
        config=make_config(),
        written_splits=(written_split,),
        output_directory=tmp_path,
    )
    second = write_finqa_manifest(
        config=make_config(),
        written_splits=(written_split,),
        output_directory=tmp_path,
    )

    assert first.checksum_sha256 == second.checksum_sha256
