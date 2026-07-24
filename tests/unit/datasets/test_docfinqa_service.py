import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import pytest

from document_rag.datasets.config import (
    DatasetConfig,
    DatasetFiles,
)
from document_rag.datasets.docfinqa.service import (
    DocFinQAProgressEvent,
    DocFinQAProgressStage,
    prepare_docfinqa_dataset,
)
from document_rag.datasets.models import (
    DatasetName,
    DatasetSplit,
)


class _PrepareDocFinQAArguments(TypedDict):
    docfinqa_source_directory: Path
    finqa_source_directory: Path
    docfinqa_config: DatasetConfig
    finqa_config: DatasetConfig
    splits: Sequence[DatasetSplit]
    chunk_size: int
    chunk_overlap: int
    evidence_minimum_score: float


def make_config(
    *,
    name: DatasetName,
) -> DatasetConfig:
    return DatasetConfig(
        name=name,
        schema_version="1",
        source_url=f"https://example.com/{name.value}",
        source_revision="a" * 40,
        files=DatasetFiles(
            train="train.json",
            validation="dev.json",
            test="test.json",
        ),
    )


def make_finqa_record(
    *,
    record_id: str = "ABC/2020/page_1.pdf-1",
    filename: str = "ABC/2020/page_1.pdf",
    question: str = "What was the revenue?",
    answer: str = "100",
    evidence: str = "revenue was 100",
) -> dict[str, object]:
    return {
        "id": record_id,
        "filename": filename,
        "pre_text": [
            "Annual report introduction.",
        ],
        "post_text": [
            "Annual report conclusion.",
        ],
        "table_ori": [
            ["Metric", "2020"],
            ["Revenue", "100"],
        ],
        "table": [
            ["Metric", "2020"],
            ["Revenue", "100"],
        ],
        "qa": {
            "id": record_id,
            "question": question,
            "answer": answer,
            "exe_ans": 100,
            "explanation": "Revenue was 100.",
            "program": "divide(200, const_2)",
            "program_re": "divide(200, const_2)",
            "gold_inds": {
                "text_1": evidence,
            },
            "steps": [],
        },
    }


def make_docfinqa_record(
    *,
    context: str = ("Annual report. Revenue was 100. End of report."),
    question: str = "What was the revenue?",
    answer: str = "100",
) -> dict[str, str]:
    return {
        "Context": context,
        "Question": question,
        "Program": "answer = 100",
        "Answer": answer,
    }


def write_json(
    path: Path,
    payload: object,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def prepare_sources(
    *,
    finqa_directory: Path,
    docfinqa_directory: Path,
) -> None:
    write_json(
        finqa_directory / "train.json",
        [make_finqa_record()],
    )
    write_json(
        docfinqa_directory / "train.json",
        [make_docfinqa_record()],
    )


def test_service_prepares_selected_split(
    tmp_path: Path,
) -> None:
    finqa_directory = tmp_path / "finqa"
    docfinqa_directory = tmp_path / "docfinqa"
    output_directory = tmp_path / "output"
    progress_events: list[DocFinQAProgressEvent] = []

    prepare_sources(
        finqa_directory=finqa_directory,
        docfinqa_directory=docfinqa_directory,
    )

    result = prepare_docfinqa_dataset(
        docfinqa_source_directory=docfinqa_directory,
        finqa_source_directory=finqa_directory,
        output_directory=output_directory,
        docfinqa_config=make_config(
            name=DatasetName.DOCFINQA,
        ),
        finqa_config=make_config(
            name=DatasetName.FINQA,
        ),
        splits=[DatasetSplit.TRAIN],
        chunk_size=100,
        chunk_overlap=20,
        evidence_minimum_score=0.8,
        progress_callback=progress_events.append,
    )

    assert len(result.splits) == 1

    split_result = result.splits[0]

    assert split_result.written_split.split is DatasetSplit.TRAIN
    assert split_result.written_split.documents.record_count == 1
    assert split_result.written_split.elements.record_count == 1
    assert split_result.written_split.examples.record_count == 1

    assert split_result.stats.total_records == 1
    assert split_result.stats.normalized_records == 1
    assert split_result.stats.exact_links == 1
    assert split_result.stats.skipped_records == 0

    assert result.manifest.path.is_file()

    manifest = json.loads(result.manifest.path.read_text(encoding="utf-8"))

    assert manifest["dataset"] == "docfinqa"
    assert list(manifest["splits"]) == ["train"]
    assert set(manifest["sources"]) == {"docfinqa", "finqa"}

    train_manifest = manifest["splits"]["train"]

    assert train_manifest["statistics"]["total_records"] == 1
    assert train_manifest["statistics"]["normalized_records"] == 1

    assert [event.stage for event in progress_events] == [
        DocFinQAProgressStage.STARTED,
        DocFinQAProgressStage.PROCESSING,
        DocFinQAProgressStage.COMPLETED,
    ]
    assert progress_events[0].stats.total_records == 0
    assert progress_events[1].stats.total_records == 1
    assert progress_events[2].stats.normalized_records == 1
    assert len(result.sample_splits) == 1

    sample_split = result.sample_splits[0]

    assert sample_split.split is DatasetSplit.TRAIN
    assert sample_split.documents.record_count == 1
    assert sample_split.elements.record_count == 1
    assert sample_split.examples.record_count == 1

    assert result.sample_manifest.path.is_file()
    assert result.integrity_report.document_overlaps == ()
    assert result.sample_integrity_report.document_overlaps == ()


def test_service_is_deterministic(
    tmp_path: Path,
) -> None:
    finqa_directory = tmp_path / "finqa"
    docfinqa_directory = tmp_path / "docfinqa"

    prepare_sources(
        finqa_directory=finqa_directory,
        docfinqa_directory=docfinqa_directory,
    )

    arguments: _PrepareDocFinQAArguments = {
        "docfinqa_source_directory": (docfinqa_directory),
        "finqa_source_directory": finqa_directory,
        "docfinqa_config": make_config(
            name=DatasetName.DOCFINQA,
        ),
        "finqa_config": make_config(
            name=DatasetName.FINQA,
        ),
        "splits": [DatasetSplit.TRAIN],
        "chunk_size": 100,
        "chunk_overlap": 20,
        "evidence_minimum_score": 0.8,
    }

    first_result = prepare_docfinqa_dataset(
        output_directory=tmp_path / "first",
        **arguments,
    )
    second_result = prepare_docfinqa_dataset(
        output_directory=tmp_path / "second",
        **arguments,
    )

    assert first_result.manifest.sha256 == second_result.manifest.sha256

    first_split = first_result.splits[0].written_split
    second_split = second_result.splits[0].written_split

    artifact_pairs = (
        (
            first_split.documents,
            second_split.documents,
        ),
        (
            first_split.elements,
            second_split.elements,
        ),
        (
            first_split.examples,
            second_split.examples,
        ),
    )

    for first_artifact, second_artifact in artifact_pairs:
        assert first_artifact.sha256 == second_artifact.sha256
        assert first_artifact.path.read_bytes() == second_artifact.path.read_bytes()

    assert first_result.sample_manifest.sha256 == second_result.sample_manifest.sha256

    first_sample = first_result.sample_splits[0]
    second_sample = second_result.sample_splits[0]

    sample_artifact_pairs = (
        (
            first_sample.documents,
            second_sample.documents,
        ),
        (
            first_sample.elements,
            second_sample.elements,
        ),
        (
            first_sample.examples,
            second_sample.examples,
        ),
    )

    for first_artifact, second_artifact in sample_artifact_pairs:
        assert first_artifact.sha256 == second_artifact.sha256


def test_service_orders_selected_splits(
    tmp_path: Path,
) -> None:
    finqa_directory = tmp_path / "finqa"
    docfinqa_directory = tmp_path / "docfinqa"

    for filename in ("train.json", "dev.json"):
        question = (
            "What was the revenue?"
            if filename == "train.json"
            else "How much revenue was reported?"
        )
        write_json(
            finqa_directory / filename,
            [
                make_finqa_record(
                    record_id=f"ABC/2020/page_1.pdf-{filename}",
                    question=question,
                )
            ],
        )
        write_json(
            docfinqa_directory / filename,
            [make_docfinqa_record(question=question)],
        )

    result = prepare_docfinqa_dataset(
        docfinqa_source_directory=docfinqa_directory,
        finqa_source_directory=finqa_directory,
        output_directory=tmp_path / "output",
        docfinqa_config=make_config(
            name=DatasetName.DOCFINQA,
        ),
        finqa_config=make_config(
            name=DatasetName.FINQA,
        ),
        splits=[
            DatasetSplit.TRAIN,
            DatasetSplit.VALIDATION,
        ],
        chunk_size=100,
        chunk_overlap=20,
        evidence_minimum_score=0.8,
    )

    assert [split_result.written_split.split for split_result in result.splits] == [
        DatasetSplit.TRAIN,
        DatasetSplit.VALIDATION,
    ]
    assert len(result.integrity_report.document_overlaps) == 1


def test_service_rejects_cross_split_example_overlap(
    tmp_path: Path,
) -> None:
    finqa_directory = tmp_path / "finqa"
    docfinqa_directory = tmp_path / "docfinqa"

    for filename in ("train.json", "dev.json"):
        write_json(
            finqa_directory / filename,
            [make_finqa_record()],
        )
        write_json(
            docfinqa_directory / filename,
            [make_docfinqa_record()],
        )

    with pytest.raises(
        ValueError,
        match="share 1 example IDs",
    ):
        prepare_docfinqa_dataset(
            docfinqa_source_directory=docfinqa_directory,
            finqa_source_directory=finqa_directory,
            output_directory=tmp_path / "output",
            docfinqa_config=make_config(
                name=DatasetName.DOCFINQA,
            ),
            finqa_config=make_config(
                name=DatasetName.FINQA,
            ),
            splits=[
                DatasetSplit.TRAIN,
                DatasetSplit.VALIDATION,
            ],
            chunk_size=100,
            chunk_overlap=20,
            evidence_minimum_score=0.8,
        )


def test_service_rejects_empty_split_selection(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="At least one DocFinQA split",
    ):
        prepare_docfinqa_dataset(
            docfinqa_source_directory=(tmp_path / "docfinqa"),
            finqa_source_directory=(tmp_path / "finqa"),
            output_directory=tmp_path / "output",
            docfinqa_config=make_config(
                name=DatasetName.DOCFINQA,
            ),
            finqa_config=make_config(
                name=DatasetName.FINQA,
            ),
            splits=[],
        )


def test_service_rejects_duplicate_splits(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="contains duplicates",
    ):
        prepare_docfinqa_dataset(
            docfinqa_source_directory=(tmp_path / "docfinqa"),
            finqa_source_directory=(tmp_path / "finqa"),
            output_directory=tmp_path / "output",
            docfinqa_config=make_config(
                name=DatasetName.DOCFINQA,
            ),
            finqa_config=make_config(
                name=DatasetName.FINQA,
            ),
            splits=[
                DatasetSplit.TRAIN,
                DatasetSplit.TRAIN,
            ],
        )


def test_service_rejects_wrong_docfinqa_config(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="requires a DocFinQA configuration",
    ):
        prepare_docfinqa_dataset(
            docfinqa_source_directory=(tmp_path / "docfinqa"),
            finqa_source_directory=(tmp_path / "finqa"),
            output_directory=tmp_path / "output",
            docfinqa_config=make_config(
                name=DatasetName.FINQA,
            ),
            finqa_config=make_config(
                name=DatasetName.FINQA,
            ),
            splits=[DatasetSplit.TRAIN],
        )


def test_service_rejects_wrong_finqa_config(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="requires a FinQA configuration",
    ):
        prepare_docfinqa_dataset(
            docfinqa_source_directory=(tmp_path / "docfinqa"),
            finqa_source_directory=(tmp_path / "finqa"),
            output_directory=tmp_path / "output",
            docfinqa_config=make_config(
                name=DatasetName.DOCFINQA,
            ),
            finqa_config=make_config(
                name=DatasetName.DOCFINQA,
            ),
            splits=[DatasetSplit.TRAIN],
        )
