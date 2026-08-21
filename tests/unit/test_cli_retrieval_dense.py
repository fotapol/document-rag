"""Tests for the dense retrieval benchmark CLI command."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from document_rag import cli
from document_rag.datasets.models import DatasetName, DatasetSplit


def test_cli_runs_dense_benchmark_with_model_configuration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Dense CLI controls should reach model and benchmark construction."""

    captured_model_arguments: dict[str, Any] = {}
    captured_benchmark_arguments: dict[str, Any] = {}
    fake_embedder = SimpleNamespace(
        model_id="custom/model",
        model_revision="revision-42",
        device="cpu",
        dimension=384,
    )

    def fake_embedder_factory(**arguments: Any) -> SimpleNamespace:
        captured_model_arguments.update(arguments)
        return fake_embedder

    def fake_benchmark(**arguments: Any) -> SimpleNamespace:
        captured_benchmark_arguments.update(arguments)
        return SimpleNamespace(
            metrics=SimpleNamespace(
                query_count=2,
                hit_rate_at_k=((1, 0.5), (5, 1.0)),
                recall_at_k=((1, 0.25), (5, 0.75)),
                mrr=0.75,
            ),
            document_count=1,
            indexed_chunk_count=4,
            embedding_cache_reused=True,
            embeddings_path=Path("cache/embeddings.npy"),
            predictions_path=Path("dense/dense_predictions.jsonl"),
            metrics_path=Path("dense/dense_metrics.json"),
            comparison_path=Path("dense/retrieval_comparison.json"),
        )

    monkeypatch.setattr(cli, "SentenceTransformerEmbedder", fake_embedder_factory)
    monkeypatch.setattr(cli, "run_dense_benchmark", fake_benchmark)

    exit_code = cli.main(
        [
            "retrieval",
            "dense",
            "--finqa",
            "processed/finqa",
            "--docfinqa",
            "processed/docfinqa",
            "--bm25-metrics",
            "bm25/retrieval_metrics.json",
            "--output",
            "dense",
            "--embedding-model",
            "custom/model",
            "--model-revision",
            "revision-42",
            "--batch-size",
            "16",
            "--device",
            "cpu",
            "--cache-directory",
            "cache",
            "--split",
            "validation",
            "--top-k",
            "1",
            "--top-k",
            "5",
        ]
    )

    assert exit_code == 0
    assert captured_model_arguments == {
        "model_id": "custom/model",
        "model_revision": "revision-42",
        "device": "cpu",
        "show_progress": True,
    }
    assert captured_benchmark_arguments == {
        "dataset_directories": {
            DatasetName.FINQA: Path("processed/finqa"),
            DatasetName.DOCFINQA: Path("processed/docfinqa"),
        },
        "output_directory": Path("dense"),
        "bm25_metrics_path": Path("bm25/retrieval_metrics.json"),
        "embedder": fake_embedder,
        "batch_size": 16,
        "split": DatasetSplit.VALIDATION,
        "cache_directory": Path("cache"),
        "k_values": (1, 5),
    }
    output = capsys.readouterr().out
    assert "Completed dense retrieval benchmark" in output
    assert "Embedding dimension: 384" in output
    assert "Embedding cache (reused)" in output
    assert "MRR: 0.750000" in output


def test_cli_rejects_dense_benchmark_without_dataset(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Dense retrieval requires at least one frozen dataset input."""

    exit_code = cli.main(
        [
            "retrieval",
            "dense",
            "--bm25-metrics",
            "bm25/retrieval_metrics.json",
            "--output",
            "dense",
        ]
    )

    assert exit_code == 1
    assert "requires --finqa and/or --docfinqa" in capsys.readouterr().err


def test_cli_reports_dense_model_or_benchmark_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Model and benchmark errors should follow the established CLI path."""

    def raise_model_error(**arguments: Any) -> None:
        del arguments
        raise RuntimeError("dense model unavailable")

    monkeypatch.setattr(cli, "SentenceTransformerEmbedder", raise_model_error)

    exit_code = cli.main(
        [
            "retrieval",
            "dense",
            "--finqa",
            "processed/finqa",
            "--bm25-metrics",
            "bm25/retrieval_metrics.json",
            "--output",
            "dense",
        ]
    )

    assert exit_code == 1
    assert "error: dense model unavailable" in capsys.readouterr().err
