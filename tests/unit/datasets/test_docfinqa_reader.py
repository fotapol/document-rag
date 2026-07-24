import json
from pathlib import Path

import pytest

from document_rag.datasets import (
    DatasetConfig,
    DatasetFiles,
    DatasetName,
    DatasetSplit,
)
from document_rag.datasets.docfinqa import (
    DocFinQARawReader,
)


def make_config(
    *,
    dataset_name: DatasetName = DatasetName.DOCFINQA,
) -> DatasetConfig:
    return DatasetConfig(
        name=dataset_name,
        schema_version="1",
        source_url=("https://huggingface.co/datasets/kensho/DocFinQA"),
        source_revision="a" * 40,
        files=DatasetFiles(
            train="train.json",
            validation="dev.json",
            test="test.json",
        ),
    )


def make_raw_record(
    *,
    context: str = "Annual report content.",
    question: str = "What was the revenue?",
    program: str = "answer = 120",
    answer: str = "120",
) -> dict[str, str]:
    return {
        "Context": context,
        "Question": question,
        "Program": program,
        "Answer": answer,
    }


def write_split(
    source_directory: Path,
    *,
    filename: str,
    records: list[dict[str, str]],
) -> None:
    source_directory.mkdir(parents=True, exist_ok=True)

    (source_directory / filename).write_text(
        json.dumps(
            records,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_reader_streams_valid_records(
    tmp_path: Path,
) -> None:
    write_split(
        tmp_path,
        filename="train.json",
        records=[
            make_raw_record(),
            make_raw_record(
                question="What was the operating income?",
                answer="40",
            ),
        ],
    )

    reader = DocFinQARawReader(
        source_directory=tmp_path,
        config=make_config(),
    )

    records = list(reader.iter_split(DatasetSplit.TRAIN))

    assert len(records) == 2
    assert records[0].context == "Annual report content."
    assert records[0].question == "What was the revenue?"
    assert records[1].answer == "40"


def test_reader_accepts_empty_program(
    tmp_path: Path,
) -> None:
    write_split(
        tmp_path,
        filename="dev.json",
        records=[
            make_raw_record(program=""),
        ],
    )

    reader = DocFinQARawReader(
        source_directory=tmp_path,
        config=make_config(),
    )

    records = list(reader.iter_split(DatasetSplit.VALIDATION))

    assert records[0].program == ""


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("Context", ""),
        ("Question", " "),
        ("Answer", ""),
    ],
)
def test_reader_rejects_empty_required_fields(
    tmp_path: Path,
    field_name: str,
    field_value: str,
) -> None:
    record = make_raw_record()
    record[field_name] = field_value

    write_split(
        tmp_path,
        filename="train.json",
        records=[record],
    )

    reader = DocFinQARawReader(
        source_directory=tmp_path,
        config=make_config(),
    )

    with pytest.raises(
        ValueError,
        match="Invalid DocFinQA record at index 0",
    ):
        list(reader.iter_split(DatasetSplit.TRAIN))


def test_reader_rejects_missing_required_field(
    tmp_path: Path,
) -> None:
    record = make_raw_record()
    del record["Answer"]

    write_split(
        tmp_path,
        filename="train.json",
        records=[record],
    )

    reader = DocFinQARawReader(
        source_directory=tmp_path,
        config=make_config(),
    )

    with pytest.raises(
        ValueError,
        match="Invalid DocFinQA record at index 0",
    ):
        list(reader.iter_split(DatasetSplit.TRAIN))


def test_reader_rejects_unknown_fields(
    tmp_path: Path,
) -> None:
    record = make_raw_record()
    record["Unexpected"] = "value"

    write_split(
        tmp_path,
        filename="train.json",
        records=[record],
    )

    reader = DocFinQARawReader(
        source_directory=tmp_path,
        config=make_config(),
    )

    with pytest.raises(
        ValueError,
        match="Invalid DocFinQA record at index 0",
    ):
        list(reader.iter_split(DatasetSplit.TRAIN))


def test_reader_rejects_empty_split(
    tmp_path: Path,
) -> None:
    write_split(
        tmp_path,
        filename="test.json",
        records=[],
    )

    reader = DocFinQARawReader(
        source_directory=tmp_path,
        config=make_config(),
    )

    with pytest.raises(
        ValueError,
        match="DocFinQA split is empty",
    ):
        list(reader.iter_split(DatasetSplit.TEST))


def test_reader_rejects_wrong_dataset_config(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="requires a DocFinQA configuration",
    ):
        DocFinQARawReader(
            source_directory=tmp_path,
            config=make_config(dataset_name=DatasetName.FINQA),
        )


def test_reader_rejects_missing_file(
    tmp_path: Path,
) -> None:
    reader = DocFinQARawReader(
        source_directory=tmp_path,
        config=make_config(),
    )

    with pytest.raises(FileNotFoundError):
        list(reader.iter_split(DatasetSplit.TRAIN))
