import json
from pathlib import Path

from document_rag.cli import main


def write_config(path: Path) -> None:
    path.write_text(
        """
name = "finqa"
schema_version = "1"
source_url = "https://github.com/example/finqa.git"
source_revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

[files]
train = "dataset/train.json"
validation = "dataset/dev.json"
test = "dataset/test.json"
""".strip(),
        encoding="utf-8",
    )


def make_raw_record() -> dict[str, object]:
    return {
        "pre_text": ["Revenue increased during the year."],
        "post_text": ["Additional information follows."],
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


def test_cli_prepares_finqa_split(
    tmp_path: Path,
    capsys: object,
) -> None:
    source_directory = tmp_path / "source"
    dataset_directory = source_directory / "dataset"
    output_directory = tmp_path / "processed"
    config_path = tmp_path / "finqa.toml"

    dataset_directory.mkdir(parents=True)

    write_config(config_path)

    (dataset_directory / "train.json").write_text(
        json.dumps([make_raw_record()]),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "data",
            "prepare",
            "--dataset",
            "finqa",
            "--config",
            str(config_path),
            "--source",
            str(source_directory),
            "--output",
            str(output_directory),
            "--split",
            "train",
        ]
    )

    assert exit_code == 0
    assert (output_directory / "train" / "documents.jsonl").is_file()
    assert (output_directory / "train" / "elements.jsonl").is_file()
    assert (output_directory / "train" / "examples.jsonl").is_file()
    assert (output_directory / "manifest.json").is_file()


def test_cli_returns_error_for_missing_source(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "finqa.toml"
    write_config(config_path)

    exit_code = main(
        [
            "data",
            "prepare",
            "--dataset",
            "finqa",
            "--config",
            str(config_path),
            "--source",
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "processed"),
            "--split",
            "train",
        ]
    )

    assert exit_code == 1
