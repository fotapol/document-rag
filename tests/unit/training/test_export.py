import json
from pathlib import Path
from typing import Any

import pytest

from document_rag.datasets.models import DatasetSplit
from document_rag.training.export import (
    SYSTEM_PROMPT,
    export_financial_qa_training_data,
)


def test_export_combines_finqa_and_docfinqa(
    tmp_path: Path,
) -> None:
    finqa = tmp_path / "finqa"
    docfinqa = tmp_path / "docfinqa"
    output = tmp_path / "training"

    _write_finqa(
        finqa,
        split=DatasetSplit.TRAIN,
        example_id="finqa-example",
    )
    _write_docfinqa(
        docfinqa,
        split=DatasetSplit.TRAIN,
        example_id="docfinqa-example",
    )

    result = export_financial_qa_training_data(
        finqa_directory=finqa,
        docfinqa_directory=docfinqa,
        output_directory=output,
        splits=[DatasetSplit.TRAIN],
    )

    assert len(result.artifacts) == 1
    artifact = result.artifacts[0]
    assert artifact.record_count == 2

    records = _read_jsonl(artifact.path)

    assert [record["dataset"] for record in records] == ["docfinqa", "finqa"]

    for record in records:
        messages = record["messages"]
        assert [message["role"] for message in messages] == ["system", "user", "assistant"]
        assert messages[0]["content"] == SYSTEM_PROMPT
        assert "Context:\n" in messages[1]["content"]
        assert "\n\nQuestion:\n" in messages[1]["content"]

    finqa_record = next(record for record in records if record["dataset"] == "finqa")
    assert "Fuel expense was $10,592." in (finqa_record["messages"][1]["content"])
    assert finqa_record["messages"][2]["content"] == "31903.6"

    docfinqa_record = next(record for record in records if record["dataset"] == "docfinqa")
    assert "Interest expense was 3.8%." in (docfinqa_record["messages"][1]["content"])
    assert docfinqa_record["messages"][2]["content"] == "380"


def test_export_is_deterministic(
    tmp_path: Path,
) -> None:
    finqa = tmp_path / "finqa"
    docfinqa = tmp_path / "docfinqa"

    _write_finqa(
        finqa,
        split=DatasetSplit.TRAIN,
        example_id="finqa-example",
    )
    _write_docfinqa(
        docfinqa,
        split=DatasetSplit.TRAIN,
        example_id="docfinqa-example",
    )

    first = export_financial_qa_training_data(
        finqa_directory=finqa,
        docfinqa_directory=docfinqa,
        output_directory=tmp_path / "first",
        splits=[DatasetSplit.TRAIN],
    )
    second = export_financial_qa_training_data(
        finqa_directory=finqa,
        docfinqa_directory=docfinqa,
        output_directory=tmp_path / "second",
        splits=[DatasetSplit.TRAIN],
    )

    assert first.artifacts[0].sha256 == second.artifacts[0].sha256
    assert first.artifacts[0].path.read_bytes() == second.artifacts[0].path.read_bytes()
    assert first.manifest_sha256 == second.manifest_sha256


def test_export_rejects_unknown_supporting_element(
    tmp_path: Path,
) -> None:
    finqa = tmp_path / "finqa"
    docfinqa = tmp_path / "docfinqa"

    _write_finqa(
        finqa,
        split=DatasetSplit.TRAIN,
        example_id="finqa-example",
        supporting_element_id="missing-element",
    )
    _write_docfinqa(
        docfinqa,
        split=DatasetSplit.TRAIN,
        example_id="docfinqa-example",
    )

    with pytest.raises(
        ValueError,
        match="unknown element",
    ):
        export_financial_qa_training_data(
            finqa_directory=finqa,
            docfinqa_directory=docfinqa,
            output_directory=tmp_path / "output",
            splits=[DatasetSplit.TRAIN],
        )


def test_export_rejects_cross_split_example_overlap(
    tmp_path: Path,
) -> None:
    finqa = tmp_path / "finqa"
    docfinqa = tmp_path / "docfinqa"

    for split in (
        DatasetSplit.TRAIN,
        DatasetSplit.TEST,
    ):
        _write_finqa(
            finqa,
            split=split,
            example_id="same-example",
        )
        _write_docfinqa(
            docfinqa,
            split=split,
            example_id=f"docfinqa-{split.value}",
        )

    with pytest.raises(
        ValueError,
        match="overlap between test and train",
    ):
        export_financial_qa_training_data(
            finqa_directory=finqa,
            docfinqa_directory=docfinqa,
            output_directory=tmp_path / "output",
            splits=[
                DatasetSplit.TRAIN,
                DatasetSplit.TEST,
            ],
        )


def _write_finqa(
    root: Path,
    *,
    split: DatasetSplit,
    example_id: str,
    supporting_element_id: str = "finqa-element",
) -> None:
    _write_manifest(root, dataset="finqa")
    split_directory = root / split.value

    _write_jsonl(
        split_directory / "elements.jsonl",
        [
            {
                "document_id": "finqa-document",
                "element_id": "finqa-element",
                "element_type": "paragraph",
                "metadata": {"dataset": "finqa"},
                "page_number": 1,
                "parent_element_id": None,
                "section": None,
                "source_text": "Fuel expense was $10,592.",
                "table_coordinates": None,
            }
        ],
    )
    _write_jsonl(
        split_directory / "examples.jsonl",
        [
            {
                "dataset": "finqa",
                "example_id": example_id,
                "question": {
                    "document_id": "finqa-document",
                    "metadata": {"dataset": "finqa"},
                    "question_id": example_id,
                    "question_type": "unknown",
                    "text": "What were total expenses?",
                },
                "reference_answer": {
                    "executable_answer": 31903.6,
                    "explanation": None,
                    "normalized_program": None,
                    "program": None,
                    "steps": [],
                    "text": "31903.6",
                },
                "split": split.value,
                "supporting_facts": [
                    {
                        "element_id": supporting_element_id,
                        "source_key": "table_1",
                    }
                ],
            }
        ],
    )


def _write_docfinqa(
    root: Path,
    *,
    split: DatasetSplit,
    example_id: str,
) -> None:
    _write_manifest(root, dataset="docfinqa")
    split_directory = root / split.value

    _write_jsonl(
        split_directory / "elements.jsonl",
        [
            {
                "document_id": "docfinqa-document",
                "element_id": "docfinqa-element",
                "element_type": "paragraph",
                "metadata": {
                    "chunk_index": 0,
                    "dataset": "docfinqa",
                    "end_char": 26,
                    "split": split.value,
                    "start_char": 0,
                },
                "page_number": 1,
                "parent_element_id": None,
                "section": None,
                "source_text": "Interest expense was 3.8%.",
                "table_coordinates": None,
            }
        ],
    )
    _write_jsonl(
        split_directory / "examples.jsonl",
        [
            {
                "dataset": "docfinqa",
                "example_id": example_id,
                "question": {
                    "document_id": "docfinqa-document",
                    "metadata": {
                        "dataset": "docfinqa",
                        "link_status": "exact",
                        "source_example_id": "finqa-source",
                        "source_file": "report.pdf",
                    },
                    "question_id": example_id,
                    "question_type": "unknown",
                    "text": "What was interest expense?",
                },
                "reference_answer": {
                    "executable_answer": None,
                    "explanation": None,
                    "normalized_program": None,
                    "program": "answer = 380",
                    "steps": [],
                    "text": "380",
                },
                "split": split.value,
                "supporting_facts": [
                    {
                        "element_id": "docfinqa-element",
                        "score": 1.0,
                        "source_key": "text_1",
                    }
                ],
            }
        ],
    )


def _write_manifest(
    root: Path,
    *,
    dataset: str,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "schema_version": "1",
                "source": {
                    "revision": "a" * 40,
                    "url": f"https://example.com/{dataset}",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
