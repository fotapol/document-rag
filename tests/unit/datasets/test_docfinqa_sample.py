import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from document_rag.datasets.config import DatasetConfig, DatasetFiles
from document_rag.datasets.docfinqa.integrity import validate_docfinqa_output
from document_rag.datasets.docfinqa.sample import create_docfinqa_sample
from document_rag.datasets.models import (
    DatasetExample,
    DatasetName,
    DatasetSplit,
    ReferenceAnswer,
    SupportingFact,
)
from document_rag.domain import (
    Document,
    DocumentElement,
    DocumentElementType,
    Question,
)


def make_config(
    name: DatasetName = DatasetName.DOCFINQA,
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


def write_source_output(
    root: Path,
    *,
    split: DatasetSplit = DatasetSplit.TRAIN,
    document_count: int = 6,
    examples_per_document: int = 2,
) -> None:
    documents: list[dict[str, Any]] = []
    elements: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    for document_number in range(document_count):
        document_id = f"docfinqa:document:{document_number:04d}"
        element_id = f"{document_id}:char-100-20:chunk-000000"
        source_text = f"Revenue for document {document_number} was 100."

        documents.append(
            Document(
                document_id=document_id,
                file_name=f"{document_number:04d}.txt",
                mime_type="text/plain",
                page_count=1,
                metadata={
                    "dataset": DatasetName.DOCFINQA.value,
                    "split": split.value,
                },
            ).model_dump(mode="json")
        )
        elements.append(
            DocumentElement(
                document_id=document_id,
                element_id=element_id,
                element_type=DocumentElementType.PARAGRAPH,
                source_text=source_text,
                page_number=1,
                metadata={
                    "dataset": DatasetName.DOCFINQA.value,
                    "split": split.value,
                    "chunk_index": 0,
                    "start_char": 0,
                    "end_char": len(source_text),
                },
            ).model_dump(mode="json")
        )

        for example_number in range(examples_per_document):
            example_id = f"docfinqa:example:{document_number:04d}:{example_number:04d}"
            examples.append(
                DatasetExample(
                    dataset=DatasetName.DOCFINQA,
                    split=split,
                    example_id=example_id,
                    question=Question(
                        question_id=example_id,
                        document_id=document_id,
                        text=f"What was revenue in document {document_number}?",
                        metadata={
                            "source_example_id": (f"report-{document_number}-{example_number}"),
                            "source_file": f"report-{document_number}.pdf",
                            "link_status": "exact",
                        },
                    ),
                    reference_answer=ReferenceAnswer(
                        text="100",
                        program="answer = 100",
                    ),
                    supporting_facts=(
                        SupportingFact(
                            element_id=element_id,
                            score=1.0,
                            source_key="text_1",
                        ),
                    ),
                ).model_dump(mode="json")
            )

    split_directory = root / split.value

    document_artifact = write_jsonl(
        root=root,
        path=split_directory / "documents.jsonl",
        records=documents,
    )
    element_artifact = write_jsonl(
        root=root,
        path=split_directory / "elements.jsonl",
        records=elements,
    )
    example_artifact = write_jsonl(
        root=root,
        path=split_directory / "examples.jsonl",
        records=examples,
    )

    manifest = {
        "dataset": "docfinqa",
        "schema_version": "1",
        "sources": {
            "docfinqa": {
                "revision": "a" * 40,
                "url": "https://example.com/docfinqa",
            },
            "finqa": {
                "revision": "a" * 40,
                "url": "https://example.com/finqa",
            },
        },
        "preprocessing": {
            "chunk_overlap": 20,
            "chunk_size": 100,
            "chunk_stride": 80,
            "evidence_minimum_score": 0.6,
        },
        "splits": {
            split.value: {
                "artifacts": {
                    "documents": document_artifact,
                    "elements": element_artifact,
                    "examples": example_artifact,
                },
                "statistics": {
                    "equivalent_links": 0,
                    "exact_links": len(examples),
                    "normalized_records": len(examples),
                    "skipped_ambiguous": 0,
                    "skipped_answer_mismatch": 0,
                    "skipped_duplicate": 0,
                    "skipped_evidence_incomplete": 0,
                    "skipped_records": 0,
                    "total_records": len(examples),
                    "unique_documents": len(documents),
                    "unmatched_evidence_facts": 0,
                },
            }
        },
    }

    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(
    *,
    root: Path,
    path: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)

    content = b"".join(
        (
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for record in records
    )
    path.write_bytes(content)

    return {
        "path": path.relative_to(root).as_posix(),
        "record_count": len(records),
        "sha256": sha256(content).hexdigest(),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_sample_selects_stable_document_limit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "sample"

    write_source_output(source)

    result = create_docfinqa_sample(
        input_directory=source,
        output_directory=output,
        config=make_config(),
        finqa_config=make_config(DatasetName.FINQA),
        splits=[DatasetSplit.TRAIN],
        chunk_size=100,
        chunk_overlap=20,
        evidence_minimum_score=0.6,
        documents_per_split=2,
    )

    written_split = result.splits[0]

    assert written_split.documents.record_count == 2
    assert written_split.elements.record_count == 2
    assert written_split.examples.record_count == 4

    all_document_ids = [f"docfinqa:document:{index:04d}" for index in range(6)]
    expected_ids = set(
        sorted(
            all_document_ids,
            key=lambda document_id: (
                sha256(document_id.encode("utf-8")).digest(),
                document_id,
            ),
        )[:2]
    )

    documents = read_jsonl(written_split.documents.path)
    elements = read_jsonl(written_split.elements.path)
    examples = read_jsonl(written_split.examples.path)

    assert {document["document_id"] for document in documents} == expected_ids
    assert {element["document_id"] for element in elements} == expected_ids
    assert {example["question"]["document_id"] for example in examples} == expected_ids


def test_sample_preserves_referential_integrity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "sample"

    write_source_output(source)

    create_docfinqa_sample(
        input_directory=source,
        output_directory=output,
        config=make_config(),
        finqa_config=make_config(DatasetName.FINQA),
        splits=[DatasetSplit.TRAIN],
        chunk_size=100,
        chunk_overlap=20,
        evidence_minimum_score=0.6,
        documents_per_split=3,
    )

    report = validate_docfinqa_output(output)

    assert len(report.splits) == 1
    assert len(report.splits[0].document_ids) == 3
    assert len(report.splits[0].element_ids) == 3
    assert len(report.splits[0].example_ids) == 6


def test_sample_manifest_records_selection(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "sample"

    write_source_output(source)

    result = create_docfinqa_sample(
        input_directory=source,
        output_directory=output,
        config=make_config(),
        finqa_config=make_config(DatasetName.FINQA),
        splits=[DatasetSplit.TRAIN],
        chunk_size=100,
        chunk_overlap=20,
        evidence_minimum_score=0.6,
        documents_per_split=2,
    )

    manifest = json.loads(result.manifest.path.read_text(encoding="utf-8"))

    assert manifest["sample"] == {
        "kind": "development_subset",
        "requested_documents_per_split": 2,
        "selection": "sha256(document_id)",
    }
    assert manifest["splits"]["train"]["statistics"]["unique_documents"] == 2


def test_sample_is_deterministic(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"

    write_source_output(source)

    first = create_docfinqa_sample(
        input_directory=source,
        output_directory=tmp_path / "first",
        config=make_config(),
        finqa_config=make_config(DatasetName.FINQA),
        splits=[DatasetSplit.TRAIN],
        chunk_size=100,
        chunk_overlap=20,
        evidence_minimum_score=0.6,
        documents_per_split=2,
    )
    second = create_docfinqa_sample(
        input_directory=source,
        output_directory=tmp_path / "second",
        config=make_config(),
        finqa_config=make_config(DatasetName.FINQA),
        splits=[DatasetSplit.TRAIN],
        chunk_size=100,
        chunk_overlap=20,
        evidence_minimum_score=0.6,
        documents_per_split=2,
    )

    assert first.manifest.sha256 == second.manifest.sha256

    artifact_pairs = (
        (
            first.splits[0].documents,
            second.splits[0].documents,
        ),
        (
            first.splits[0].elements,
            second.splits[0].elements,
        ),
        (
            first.splits[0].examples,
            second.splits[0].examples,
        ),
    )

    for first_artifact, second_artifact in artifact_pairs:
        assert first_artifact.sha256 == second_artifact.sha256
        assert first_artifact.path.read_bytes() == second_artifact.path.read_bytes()


def test_sample_uses_all_documents_when_limit_is_larger(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "sample"

    write_source_output(source, document_count=2)

    result = create_docfinqa_sample(
        input_directory=source,
        output_directory=output,
        config=make_config(),
        finqa_config=make_config(DatasetName.FINQA),
        splits=[DatasetSplit.TRAIN],
        chunk_size=100,
        chunk_overlap=20,
        evidence_minimum_score=0.6,
        documents_per_split=5,
    )

    assert result.splits[0].documents.record_count == 2


@pytest.mark.parametrize("documents_per_split", [0, -1])
def test_sample_rejects_invalid_document_limit(
    tmp_path: Path,
    documents_per_split: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="Sample documents per split must be positive",
    ):
        create_docfinqa_sample(
            input_directory=tmp_path,
            output_directory=tmp_path / "sample",
            config=make_config(),
            finqa_config=make_config(DatasetName.FINQA),
            splits=[DatasetSplit.TRAIN],
            chunk_size=100,
            chunk_overlap=20,
            evidence_minimum_score=0.6,
            documents_per_split=documents_per_split,
        )
