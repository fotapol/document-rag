import json
from pathlib import Path

import pytest

from document_rag.datasets.config import (
    DatasetConfig,
    DatasetFiles,
)
from document_rag.datasets.docfinqa.manifest import (
    DocFinQASplitManifestInput,
    write_docfinqa_manifest,
)
from document_rag.datasets.docfinqa.preparer import (
    DocFinQAPreparationStats,
)
from document_rag.datasets.docfinqa.writer import (
    WrittenDocFinQAArtifact,
    WrittenDocFinQASplit,
)
from document_rag.datasets.models import (
    DatasetName,
    DatasetSplit,
)


def make_config(
    *,
    name: DatasetName = DatasetName.DOCFINQA,
) -> DatasetConfig:
    return DatasetConfig(
        name=name,
        schema_version="1",
        source_url=("https://huggingface.co/datasets/kensho/DocFinQA"),
        source_revision="a" * 40,
        files=DatasetFiles(
            train="train.json",
            validation="dev.json",
            test="test.json",
        ),
    )


def make_artifact(
    *,
    path: Path,
    record_count: int,
    checksum_character: str,
) -> WrittenDocFinQAArtifact:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        "{}\n",
        encoding="utf-8",
    )

    return WrittenDocFinQAArtifact(
        path=path,
        record_count=record_count,
        sha256=checksum_character * 64,
    )


def make_split_input(
    output_directory: Path,
    *,
    split: DatasetSplit = DatasetSplit.TRAIN,
    total_records: int = 12,
    normalized_records: int = 10,
    unique_documents: int = 4,
    skipped_duplicate: int = 0,
) -> DocFinQASplitManifestInput:
    split_directory = output_directory / split.value

    written_split = WrittenDocFinQASplit(
        split=split,
        documents=make_artifact(
            path=split_directory / "documents.jsonl",
            record_count=unique_documents,
            checksum_character="a",
        ),
        elements=make_artifact(
            path=split_directory / "elements.jsonl",
            record_count=20,
            checksum_character="b",
        ),
        examples=make_artifact(
            path=split_directory / "examples.jsonl",
            record_count=normalized_records,
            checksum_character="c",
        ),
    )

    stats = DocFinQAPreparationStats(
        total_records=total_records,
        normalized_records=normalized_records,
        unique_documents=unique_documents,
        exact_links=9,
        equivalent_links=2 + skipped_duplicate,
        skipped_ambiguous=1,
        skipped_answer_mismatch=0,
        skipped_evidence_incomplete=1,
        unmatched_evidence_facts=1,
        skipped_duplicate=skipped_duplicate,
    )

    return DocFinQASplitManifestInput(
        written_split=written_split,
        stats=stats,
    )


def test_manifest_contains_source_artifacts_and_stats(
    tmp_path: Path,
) -> None:
    split_input = make_split_input(
        tmp_path,
        total_records=13,
        skipped_duplicate=1,
    )

    result = write_docfinqa_manifest(
        output_directory=tmp_path,
        config=make_config(),
        split_inputs=[split_input],
        chunk_size=2_750,
        chunk_overlap=550,
        evidence_minimum_score=0.6,
    )

    payload = json.loads(result.path.read_text(encoding="utf-8"))

    assert payload["dataset"] == "docfinqa"
    assert payload["schema_version"] == "1"

    assert payload["source"]["revision"] == "a" * 40

    assert payload["preprocessing"] == {
        "chunk_overlap": 550,
        "chunk_size": 2_750,
        "chunk_stride": 2_200,
        "evidence_minimum_score": 0.6,
    }

    train = payload["splits"]["train"]

    assert train["artifacts"]["documents"]["record_count"] == 4
    assert train["artifacts"]["elements"]["record_count"] == 20
    assert train["artifacts"]["examples"]["record_count"] == 10

    assert train["statistics"]["skipped_records"] == 3
    assert train["statistics"]["skipped_duplicate"] == 1
    assert len(result.sha256) == 64


def test_manifest_is_deterministic(
    tmp_path: Path,
) -> None:
    train_input = make_split_input(
        tmp_path,
        split=DatasetSplit.TRAIN,
    )
    validation_input = make_split_input(
        tmp_path,
        split=DatasetSplit.VALIDATION,
    )

    first_result = write_docfinqa_manifest(
        output_directory=tmp_path,
        config=make_config(),
        split_inputs=[
            validation_input,
            train_input,
        ],
        chunk_size=2_750,
        chunk_overlap=550,
        evidence_minimum_score=0.6,
    )

    first_content = first_result.path.read_bytes()

    second_result = write_docfinqa_manifest(
        output_directory=tmp_path,
        config=make_config(),
        split_inputs=[
            train_input,
            validation_input,
        ],
        chunk_size=2_750,
        chunk_overlap=550,
        evidence_minimum_score=0.6,
    )

    assert second_result.path.read_bytes() == first_content
    assert second_result.sha256 == first_result.sha256


def test_manifest_rejects_duplicate_splits(
    tmp_path: Path,
) -> None:
    split_input = make_split_input(tmp_path)

    with pytest.raises(
        ValueError,
        match="duplicate splits",
    ):
        write_docfinqa_manifest(
            output_directory=tmp_path,
            config=make_config(),
            split_inputs=[
                split_input,
                split_input,
            ],
            chunk_size=2_750,
            chunk_overlap=550,
            evidence_minimum_score=0.6,
        )


def test_manifest_rejects_inconsistent_example_count(
    tmp_path: Path,
) -> None:
    split_input = make_split_input(
        tmp_path,
        normalized_records=10,
    )

    invalid_written_split = WrittenDocFinQASplit(
        split=split_input.written_split.split,
        documents=split_input.written_split.documents,
        elements=split_input.written_split.elements,
        examples=WrittenDocFinQAArtifact(
            path=split_input.written_split.examples.path,
            record_count=9,
            sha256=(split_input.written_split.examples.sha256),
        ),
    )

    invalid_input = DocFinQASplitManifestInput(
        written_split=invalid_written_split,
        stats=split_input.stats,
    )

    with pytest.raises(
        ValueError,
        match="written examples",
    ):
        write_docfinqa_manifest(
            output_directory=tmp_path,
            config=make_config(),
            split_inputs=[invalid_input],
            chunk_size=2_750,
            chunk_overlap=550,
            evidence_minimum_score=0.6,
        )


def test_manifest_rejects_artifact_outside_output(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "output"
    split_input = make_split_input(output_directory)

    outside_path = tmp_path / "outside" / "examples.jsonl"

    outside_examples = make_artifact(
        path=outside_path,
        record_count=10,
        checksum_character="d",
    )

    invalid_input = DocFinQASplitManifestInput(
        written_split=WrittenDocFinQASplit(
            split=DatasetSplit.TRAIN,
            documents=(split_input.written_split.documents),
            elements=split_input.written_split.elements,
            examples=outside_examples,
        ),
        stats=split_input.stats,
    )

    with pytest.raises(
        ValueError,
        match="inside the output directory",
    ):
        write_docfinqa_manifest(
            output_directory=output_directory,
            config=make_config(),
            split_inputs=[invalid_input],
            chunk_size=2_750,
            chunk_overlap=550,
            evidence_minimum_score=0.6,
        )


def test_manifest_rejects_wrong_dataset_config(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="requires a DocFinQA configuration",
    ):
        write_docfinqa_manifest(
            output_directory=tmp_path,
            config=make_config(name=DatasetName.FINQA),
            split_inputs=[make_split_input(tmp_path)],
            chunk_size=2_750,
            chunk_overlap=550,
            evidence_minimum_score=0.6,
        )


def test_manifest_rejects_empty_split_list(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="At least one DocFinQA split",
    ):
        write_docfinqa_manifest(
            output_directory=tmp_path,
            config=make_config(),
            split_inputs=[],
            chunk_size=2_750,
            chunk_overlap=550,
            evidence_minimum_score=0.6,
        )


def test_manifest_records_sample_configuration(
    tmp_path: Path,
) -> None:
    split_input = make_split_input(tmp_path)

    result = write_docfinqa_manifest(
        output_directory=tmp_path,
        config=make_config(),
        split_inputs=[split_input],
        chunk_size=2_750,
        chunk_overlap=550,
        evidence_minimum_score=0.6,
        sample_documents_per_split=5,
    )

    payload = json.loads(result.path.read_text(encoding="utf-8"))

    assert payload["sample"] == {
        "kind": "development_subset",
        "requested_documents_per_split": 5,
        "selection": "sha256(document_id)",
    }
