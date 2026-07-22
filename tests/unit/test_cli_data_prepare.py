from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from document_rag import cli
from document_rag.datasets.docfinqa.service import DocFinQAProgressStage
from document_rag.datasets.models import DatasetSplit


def make_written_split(
    *,
    split: DatasetSplit = DatasetSplit.TRAIN,
) -> SimpleNamespace:
    return SimpleNamespace(
        split=split,
        documents=SimpleNamespace(
            record_count=2,
        ),
        elements=SimpleNamespace(
            record_count=20,
        ),
        examples=SimpleNamespace(
            record_count=5,
        ),
    )


def test_cli_prepares_finqa(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured_arguments: dict[str, Any] = {}
    config = object()

    monkeypatch.setattr(
        cli,
        "load_dataset_config",
        lambda path: config,
    )

    def fake_prepare_finqa_dataset(
        **arguments: Any,
    ) -> SimpleNamespace:
        captured_arguments.update(arguments)

        return SimpleNamespace(
            splits=(make_written_split(),),
            manifest=SimpleNamespace(
                path=Path("data/processed/finqa/manifest.json"),
            ),
            sample_splits=(),
            sample_manifest=None,
            report_overlaps=(),
        )

    monkeypatch.setattr(
        cli,
        "prepare_finqa_dataset",
        fake_prepare_finqa_dataset,
    )

    exit_code = cli.main(
        [
            "data",
            "prepare",
            "--dataset",
            "finqa",
            "--config",
            "configs/datasets/finqa.toml",
            "--source",
            "source/finqa",
            "--output",
            "data/processed/finqa",
            "--split",
            "train",
        ]
    )

    assert exit_code == 0

    assert captured_arguments == {
        "config": config,
        "source_directory": Path("source/finqa"),
        "output_directory": Path("data/processed/finqa"),
        "splits": (DatasetSplit.TRAIN,),
    }

    captured = capsys.readouterr()

    assert "Prepared dataset: finqa" in captured.out
    assert "2 documents" in captured.out
    assert "20 elements" in captured.out
    assert "5 examples" in captured.out


def test_cli_prepares_docfinqa(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured_arguments: dict[str, Any] = {}
    progress_times = iter((0.0, 6.0))

    monkeypatch.setattr(
        cli,
        "monotonic",
        lambda: next(progress_times),
    )

    docfinqa_config = object()
    finqa_config = object()

    def fake_load_config(
        path: Path,
    ) -> object:
        if path.name == "docfinqa.toml":
            return docfinqa_config

        return finqa_config

    monkeypatch.setattr(
        cli,
        "load_dataset_config",
        fake_load_config,
    )

    def fake_prepare_docfinqa_dataset(
        **arguments: Any,
    ) -> SimpleNamespace:
        captured_arguments.update(arguments)

        written_split = make_written_split()

        stats = SimpleNamespace(
            total_records=7,
            normalized_records=5,
            skipped_records=2,
            skipped_ambiguous=1,
            skipped_answer_mismatch=1,
            skipped_evidence_incomplete=0,
            skipped_duplicate=0,
        )

        progress_callback = arguments["progress_callback"]
        progress_callback(
            SimpleNamespace(
                split=DatasetSplit.TRAIN,
                stage=DocFinQAProgressStage.STARTED,
                stats=stats,
            )
        )
        progress_callback(
            SimpleNamespace(
                split=DatasetSplit.TRAIN,
                stage=DocFinQAProgressStage.PROCESSING,
                stats=stats,
            )
        )
        progress_callback(
            SimpleNamespace(
                split=DatasetSplit.TRAIN,
                stage=DocFinQAProgressStage.COMPLETED,
                stats=stats,
            )
        )

        return SimpleNamespace(
            splits=(
                SimpleNamespace(
                    written_split=written_split,
                    stats=stats,
                ),
            ),
            manifest=SimpleNamespace(
                path=Path("data/processed/docfinqa/manifest.json"),
            ),
        )

    monkeypatch.setattr(
        cli,
        "prepare_docfinqa_dataset",
        fake_prepare_docfinqa_dataset,
    )

    exit_code = cli.main(
        [
            "data",
            "prepare",
            "--dataset",
            "docfinqa",
            "--config",
            "configs/datasets/docfinqa.toml",
            "--source",
            "source/docfinqa",
            "--finqa-config",
            "configs/datasets/finqa.toml",
            "--finqa-source",
            "source/finqa",
            "--output",
            "data/processed/docfinqa",
            "--split",
            "train",
            "--chunk-size",
            "1000",
            "--chunk-overlap",
            "200",
            "--evidence-minimum-score",
            "0.75",
        ]
    )

    assert exit_code == 0

    progress_callback = captured_arguments.pop("progress_callback")

    assert callable(progress_callback)
    assert captured_arguments == {
        "docfinqa_source_directory": Path("source/docfinqa"),
        "finqa_source_directory": Path("source/finqa"),
        "output_directory": Path("data/processed/docfinqa"),
        "docfinqa_config": docfinqa_config,
        "finqa_config": finqa_config,
        "splits": (DatasetSplit.TRAIN,),
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "evidence_minimum_score": 0.75,
    }

    captured = capsys.readouterr()

    assert "Prepared dataset: docfinqa" in captured.out
    assert "skipped: 2 total" in captured.out
    assert "1 ambiguous" in captured.out
    assert "1 answer mismatch" in captured.out
    assert "0 duplicates" in captured.out
    assert "Preparing DocFinQA splits: train" in captured.err
    assert "[train] Starting preparation" in captured.err
    assert "[train] 7 records processed" in captured.err
    assert "[train] Completed: 7 records processed" in captured.err


@pytest.mark.parametrize(
    "missing_argument",
    [
        "--finqa-config",
        "--finqa-source",
    ],
)
def test_cli_requires_finqa_inputs_for_docfinqa(
    missing_argument: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = [
        "data",
        "prepare",
        "--dataset",
        "docfinqa",
        "--config",
        "configs/datasets/docfinqa.toml",
        "--source",
        "source/docfinqa",
        "--finqa-config",
        "configs/datasets/finqa.toml",
        "--finqa-source",
        "source/finqa",
        "--output",
        "data/processed/docfinqa",
    ]

    argument_index = arguments.index(missing_argument)
    del arguments[argument_index : argument_index + 2]

    exit_code = cli.main(arguments)

    assert exit_code == 1

    captured = capsys.readouterr()

    assert missing_argument in captured.err


def test_cli_reports_preparation_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def raise_config_error(
        path: Path,
    ) -> object:
        raise ValueError("Invalid dataset config")

    monkeypatch.setattr(
        cli,
        "load_dataset_config",
        raise_config_error,
    )

    exit_code = cli.main(
        [
            "data",
            "prepare",
            "--dataset",
            "finqa",
            "--config",
            "invalid.toml",
            "--source",
            "source/finqa",
            "--output",
            "output",
        ]
    )

    assert exit_code == 1

    captured = capsys.readouterr()

    assert "error: Invalid dataset config" in captured.err


def test_cli_uses_all_splits_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_splits: (
        tuple[
            DatasetSplit,
            ...,
        ]
        | None
    ) = None

    monkeypatch.setattr(
        cli,
        "load_dataset_config",
        lambda path: object(),
    )

    def fake_prepare_finqa_dataset(
        **arguments: Any,
    ) -> SimpleNamespace:
        nonlocal captured_splits

        captured_splits = arguments["splits"]

        return SimpleNamespace(
            splits=(),
            manifest=SimpleNamespace(
                path=Path("manifest.json"),
            ),
            sample_splits=(),
            sample_manifest=None,
            report_overlaps=(),
        )

    monkeypatch.setattr(
        cli,
        "prepare_finqa_dataset",
        fake_prepare_finqa_dataset,
    )

    exit_code = cli.main(
        [
            "data",
            "prepare",
            "--dataset",
            "finqa",
            "--config",
            "finqa.toml",
            "--source",
            "source",
            "--output",
            "output",
        ]
    )

    assert exit_code == 0
    assert captured_splits == tuple(DatasetSplit)
