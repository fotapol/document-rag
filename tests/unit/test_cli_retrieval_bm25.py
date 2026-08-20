"""Tests for the BM25 benchmark CLI command."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from document_rag import cli
from document_rag.datasets.models import DatasetName, DatasetSplit


def test_cli_runs_bm25_benchmark(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI arguments should reach the benchmark service unchanged."""

    captured_arguments: dict[str, Any] = {}

    def fake_benchmark(**arguments: Any) -> SimpleNamespace:
        captured_arguments.update(arguments)
        return SimpleNamespace(
            metrics=SimpleNamespace(
                query_count=2,
                hit_rate_at_k=((1, 0.5), (3, 1.0), (5, 1.0)),
                recall_at_k=((1, 0.25), (3, 0.75), (5, 1.0)),
                mrr=0.75,
            ),
            document_count=1,
            indexed_chunk_count=4,
            predictions_path=Path("benchmark/retrieval_predictions.jsonl"),
            metrics_path=Path("benchmark/retrieval_metrics.json"),
        )

    monkeypatch.setattr(cli, "run_bm25_benchmark", fake_benchmark)

    exit_code = cli.main(
        [
            "retrieval",
            "bm25",
            "--finqa",
            "processed/finqa",
            "--output",
            "benchmark",
            "--split",
            "validation",
            "--top-k",
            "1",
            "--top-k",
            "5",
        ]
    )

    assert exit_code == 0
    assert captured_arguments == {
        "dataset_directories": {DatasetName.FINQA: Path("processed/finqa")},
        "output_directory": Path("benchmark"),
        "split": DatasetSplit.VALIDATION,
        "k_values": (1, 5),
    }
    output = capsys.readouterr().out
    assert "Completed BM25 retrieval benchmark" in output
    assert "Hit Rate@1: 0.500000" in output
    assert "MRR: 0.750000" in output


def test_cli_rejects_benchmark_without_dataset(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """At least one normalized input directory is required."""

    exit_code = cli.main(
        [
            "retrieval",
            "bm25",
            "--output",
            "benchmark",
        ]
    )

    assert exit_code == 1
    assert "requires --finqa and/or --docfinqa" in capsys.readouterr().err


def test_cli_reports_benchmark_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Input and evaluation errors should use the established CLI error path."""

    def raise_benchmark_error(**arguments: Any) -> None:
        del arguments
        raise ValueError("invalid retrieval input")

    monkeypatch.setattr(cli, "run_bm25_benchmark", raise_benchmark_error)

    exit_code = cli.main(
        [
            "retrieval",
            "bm25",
            "--docfinqa",
            "processed/docfinqa",
            "--output",
            "benchmark",
        ]
    )

    assert exit_code == 1
    assert "error: invalid retrieval input" in capsys.readouterr().err
