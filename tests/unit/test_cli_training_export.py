from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from document_rag import cli
from document_rag.datasets.models import DatasetSplit


def test_cli_exports_training_data(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured_arguments: dict[str, Any] = {}

    def fake_export(**arguments: Any) -> SimpleNamespace:
        captured_arguments.update(arguments)
        return SimpleNamespace(
            artifacts=(
                SimpleNamespace(
                    split=DatasetSplit.TRAIN,
                    record_count=42,
                    path=Path("training/train.jsonl"),
                ),
            ),
            manifest_path=Path("training/manifest.json"),
        )

    monkeypatch.setattr(
        cli,
        "export_financial_qa_training_data",
        fake_export,
    )

    exit_code = cli.main(
        [
            "training",
            "export",
            "--finqa",
            "processed/finqa",
            "--docfinqa",
            "processed/docfinqa",
            "--output",
            "training",
            "--split",
            "train",
        ]
    )

    assert exit_code == 0
    assert captured_arguments == {
        "finqa_directory": Path("processed/finqa"),
        "docfinqa_directory": Path("processed/docfinqa"),
        "output_directory": Path("training"),
        "splits": (DatasetSplit.TRAIN,),
    }

    captured = capsys.readouterr()
    assert "42 examples" in captured.out
    assert str(Path("training/manifest.json")) in captured.out


def test_cli_reports_training_export_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def raise_export_error(**arguments: Any) -> None:
        raise ValueError("invalid training input")

    monkeypatch.setattr(
        cli,
        "export_financial_qa_training_data",
        raise_export_error,
    )

    exit_code = cli.main(
        [
            "training",
            "export",
            "--finqa",
            "processed/finqa",
            "--docfinqa",
            "processed/docfinqa",
            "--output",
            "training",
        ]
    )

    assert exit_code == 1
    assert "error: invalid training input" in capsys.readouterr().err
