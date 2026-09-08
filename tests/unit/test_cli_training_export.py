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


def test_cli_exports_rag_aligned_training_data(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The RAG command should pass generation policy and report audit counters."""

    captured_arguments: dict[str, Any] = {}

    def fake_export(**arguments: Any) -> SimpleNamespace:
        captured_arguments.update(arguments)
        return SimpleNamespace(
            artifacts=(
                SimpleNamespace(
                    split=DatasetSplit.TRAIN,
                    record_count=100,
                    supported_count=80,
                    refusal_count=20,
                    calculation_supervised_count=50,
                    context_trimmed_count=7,
                    oracle_augmented_count=12,
                    ambiguous_unit_exclusion_count=3,
                    gold_source_overflow_exclusion_count=2,
                    sequence_overflow_exclusion_count=1,
                    max_sequence_token_count=4090,
                    path=Path("rag-training/train.jsonl"),
                ),
            ),
            manifest_path=Path("rag-training/manifest.json"),
        )

    monkeypatch.setattr(cli, "export_rag_training_data", fake_export)

    exit_code = cli.main(
        [
            "training",
            "export-rag",
            "--finqa",
            "processed/finqa",
            "--docfinqa",
            "processed/docfinqa",
            "--output",
            "rag-training",
            "--split",
            "train",
            "--refusal-ratio",
            "0.2",
        ]
    )

    assert exit_code == 0
    assert captured_arguments["finqa_directory"] == Path("processed/finqa")
    assert captured_arguments["docfinqa_directory"] == Path("processed/docfinqa")
    assert captured_arguments["output_directory"] == Path("rag-training")
    assert captured_arguments["splits"] == (DatasetSplit.TRAIN,)
    assert captured_arguments["export_config"].refusal_ratio == 0.2
    assert captured_arguments["export_config"].oracle_augment is True
    assert captured_arguments["export_config"].document_resplit is True
    assert captured_arguments["export_config"].max_sequence_tokens == 4096
    assert captured_arguments["rag_config"].top_k == 5

    output = capsys.readouterr().out
    assert "80 supported, 20 refusals" in output
    assert "50 calculation supervised" in output
    assert "7 context trimmed" in output
    assert "12 oracle augmented" in output
    assert "3 ambiguous-unit exclusions" in output
    assert "2 gold-source overflow exclusions" in output
    assert "1 sequence overflow exclusions" in output
    assert "max 4090 tokens" in output


def test_cli_reports_rag_training_policy_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Invalid refusal ratios should fail before loading an embedding model."""

    monkeypatch.setattr(
        cli,
        "export_rag_training_data",
        lambda **arguments: pytest.fail(f"export should not run: {arguments}"),
    )

    exit_code = cli.main(
        [
            "training",
            "export-rag",
            "--finqa",
            "processed/finqa",
            "--docfinqa",
            "processed/docfinqa",
            "--output",
            "rag-training",
            "--refusal-ratio",
            "0.5",
        ]
    )

    assert exit_code == 1
    assert "refusal_ratio" in capsys.readouterr().err
