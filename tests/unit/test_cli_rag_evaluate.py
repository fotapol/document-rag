"""Tests for the frozen-context RAG comparison CLI."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from document_rag import cli
from document_rag.rag.errors import RAGEvaluationError


def test_cli_runs_paired_rag_evaluation_with_overrides(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI model overrides should reach the paired generator and evaluator."""

    captured: dict[str, Any] = {}
    fake_generator = object()

    def fake_generator_factory(config: object) -> object:
        captured["generator_config"] = config
        return fake_generator

    def fake_evaluation(**arguments: Any) -> SimpleNamespace:
        captured["evaluation"] = arguments
        return SimpleNamespace(
            suite=SimpleNamespace(name="diagnostic", cases=(object(), object())),
            base_metrics=SimpleNamespace(overall=SimpleNamespace(correct=2, count=2, accuracy=1.0)),
            adapter_metrics=SimpleNamespace(
                overall=SimpleNamespace(correct=1, count=2, accuracy=0.5)
            ),
            predictions_path=Path("output/rag_predictions.jsonl"),
            metrics_path=Path("output/rag_metrics.json"),
        )

    monkeypatch.setattr(cli, "QwenBaseLoraComparisonGenerator", fake_generator_factory)
    monkeypatch.setattr(cli, "run_frozen_rag_evaluation", fake_evaluation)

    exit_code = cli.main(
        [
            "rag",
            "evaluate",
            "--suite",
            "suite.json",
            "--output",
            "output",
            "--base-model",
            "example/base",
            "--base-revision",
            "base-sha",
            "--adapter-model",
            "example/adapter",
            "--adapter-revision",
            "adapter-sha",
            "--device-map",
            "cuda",
            "--max-input-tokens",
            "2048",
            "--max-new-tokens",
            "64",
        ]
    )

    assert exit_code == 0
    config = captured["generator_config"]
    assert config.base_model_id == "example/base"
    assert config.base_model_revision == "base-sha"
    assert config.adapter_model_id == "example/adapter"
    assert config.adapter_model_revision == "adapter-sha"
    assert config.generation_device_map == "cuda"
    assert config.max_input_tokens == 2048
    assert config.max_new_tokens == 64
    assert captured["evaluation"] == {
        "suite_path": Path("suite.json"),
        "output_directory": Path("output"),
        "generator": fake_generator,
        "config": config,
    }
    output = capsys.readouterr().out
    assert "Completed frozen-context RAG evaluation" in output
    assert "Base overall: 2/2 (100.00%)" in output
    assert "Adapter overall: 1/2 (50.00%)" in output


def test_cli_reports_suite_or_generation_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Evaluation failures should use the existing user-facing CLI error path."""

    def fail_evaluation(**arguments: Any) -> None:
        del arguments
        raise RAGEvaluationError("invalid frozen suite")

    monkeypatch.setattr(cli, "run_frozen_rag_evaluation", fail_evaluation)

    exit_code = cli.main(
        [
            "rag",
            "evaluate",
            "--suite",
            "suite.json",
            "--output",
            "output",
        ]
    )

    assert exit_code == 1
    assert "error: invalid frozen suite" in capsys.readouterr().err
