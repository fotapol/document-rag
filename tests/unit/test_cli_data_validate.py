from pathlib import Path
from types import SimpleNamespace

import pytest

from document_rag import cli
from document_rag.datasets.models import DatasetSplit


def test_cli_validates_docfinqa(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured_input: Path | None = None

    def fake_validate_docfinqa_output(
        input_directory: Path,
    ) -> SimpleNamespace:
        nonlocal captured_input
        captured_input = input_directory

        return SimpleNamespace(
            splits=(
                SimpleNamespace(
                    split=DatasetSplit.TEST,
                    document_ids=frozenset({"document-test"}),
                    element_ids=frozenset(
                        {
                            "element-test-1",
                            "element-test-2",
                        }
                    ),
                    example_ids=frozenset({"example-test"}),
                ),
                SimpleNamespace(
                    split=DatasetSplit.TRAIN,
                    document_ids=frozenset(
                        {
                            "document-train-1",
                            "document-train-2",
                        }
                    ),
                    element_ids=frozenset(
                        {
                            "element-train-1",
                            "element-train-2",
                            "element-train-3",
                        }
                    ),
                    example_ids=frozenset(
                        {
                            "example-train-1",
                            "example-train-2",
                        }
                    ),
                ),
            ),
            document_overlaps=(
                SimpleNamespace(
                    left_split=DatasetSplit.TEST,
                    right_split=DatasetSplit.TRAIN,
                    count=1,
                ),
            ),
        )

    monkeypatch.setattr(
        cli,
        "validate_docfinqa_output",
        fake_validate_docfinqa_output,
    )

    exit_code = cli.main(
        [
            "data",
            "validate",
            "--dataset",
            "docfinqa",
            "--input",
            "data/processed/docfinqa",
        ]
    )

    assert exit_code == 0
    assert captured_input == Path("data/processed/docfinqa")

    captured = capsys.readouterr()

    assert "Validated dataset: docfinqa" in captured.out
    assert "test: 1 documents, 2 elements, 1 examples" in captured.out
    assert "train: 2 documents, 3 elements, 2 examples" in captured.out
    assert "Warning: 1 documents are shared between test and train" in captured.err


def test_cli_returns_error_for_invalid_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def raise_integrity_error(
        input_directory: Path,
    ) -> None:
        raise ValueError("train elements SHA-256 mismatch")

    monkeypatch.setattr(
        cli,
        "validate_docfinqa_output",
        raise_integrity_error,
    )

    exit_code = cli.main(
        [
            "data",
            "validate",
            "--dataset",
            "docfinqa",
            "--input",
            "data/processed/docfinqa",
        ]
    )

    assert exit_code == 1

    captured = capsys.readouterr()

    assert "error: train elements SHA-256 mismatch" in captured.err


def test_cli_returns_error_when_manifest_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_manifest = Path("missing/manifest.json")

    def raise_missing_file(
        input_directory: Path,
    ) -> None:
        raise FileNotFoundError(missing_manifest)

    monkeypatch.setattr(
        cli,
        "validate_docfinqa_output",
        raise_missing_file,
    )

    exit_code = cli.main(
        [
            "data",
            "validate",
            "--dataset",
            "docfinqa",
            "--input",
            "missing",
        ]
    )

    assert exit_code == 1

    captured = capsys.readouterr()

    assert "error:" in captured.err
    assert "manifest.json" in captured.err
