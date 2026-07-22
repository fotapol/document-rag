import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from document_rag.datasets.docfinqa.integrity import (
    validate_docfinqa_output,
)


def _write_artifact(
    *,
    root: Path,
    relative_path: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    path = root / relative_path
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        "path": relative_path,
        "record_count": len(records),
        "sha256": sha256(content).hexdigest(),
    }


def _make_split(
    *,
    root: Path,
    split: str,
    document_id: str | None = None,
    example_id: str | None = None,
    supporting_element_id: str | None = None,
    element_end_char: int | None = None,
) -> dict[str, Any]:
    resolved_document_id = document_id or f"docfinqa:document:{split}"
    element_id = f"{resolved_document_id}:char-2750-550:chunk-000000"
    resolved_example_id = example_id or f"docfinqa:example:{split}"

    source_text = "Revenue was 100."
    end_char = element_end_char if element_end_char is not None else len(source_text)

    documents = [
        {
            "dataset": "docfinqa",
            "document_id": resolved_document_id,
            "split": split,
        }
    ]

    elements = [
        {
            "document_id": resolved_document_id,
            "element_id": element_id,
            "end_char": end_char,
            "index": 0,
            "source_text": source_text,
            "start_char": 0,
        }
    ]

    examples = [
        {
            "answer": "100",
            "dataset": "docfinqa",
            "document_id": resolved_document_id,
            "example_id": resolved_example_id,
            "finqa_id": "ABC/2020/page_1.pdf-1",
            "finqa_source_file": ("ABC/2020/page_1.pdf"),
            "link_status": "exact",
            "program": "answer = 100",
            "question": "What was the revenue?",
            "split": split,
            "supporting_facts": [
                {
                    "element_id": (supporting_element_id or element_id),
                    "score": 1.0,
                    "source_key": "text_1",
                }
            ],
        }
    ]

    artifacts = {
        "documents": _write_artifact(
            root=root,
            relative_path=(f"{split}/documents.jsonl"),
            records=documents,
        ),
        "elements": _write_artifact(
            root=root,
            relative_path=(f"{split}/elements.jsonl"),
            records=elements,
        ),
        "examples": _write_artifact(
            root=root,
            relative_path=(f"{split}/examples.jsonl"),
            records=examples,
        ),
    }

    return {
        "artifacts": artifacts,
        "statistics": {
            "equivalent_links": 0,
            "exact_links": 1,
            "normalized_records": 1,
            "skipped_ambiguous": 0,
            "skipped_answer_mismatch": 0,
            "skipped_duplicate": 0,
            "skipped_evidence_incomplete": 0,
            "skipped_records": 0,
            "total_records": 1,
            "unique_documents": 1,
            "unmatched_evidence_facts": 0,
        },
    }


def _write_manifest(
    *,
    root: Path,
    splits: dict[str, dict[str, Any]],
) -> None:
    payload = {
        "dataset": "docfinqa",
        "schema_version": "1",
        "splits": splits,
    }

    root.mkdir(
        parents=True,
        exist_ok=True,
    )
    (root / "manifest.json").write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_integrity_validates_complete_output(
    tmp_path: Path,
) -> None:
    train = _make_split(
        root=tmp_path,
        split="train",
    )
    _write_manifest(
        root=tmp_path,
        splits={"train": train},
    )

    report = validate_docfinqa_output(tmp_path)

    assert len(report.splits) == 1
    assert report.document_overlaps == ()

    snapshot = report.splits[0]

    assert snapshot.split.value == "train"
    assert len(snapshot.document_ids) == 1
    assert len(snapshot.element_ids) == 1
    assert len(snapshot.example_ids) == 1


def test_integrity_reports_shared_documents(
    tmp_path: Path,
) -> None:
    shared_document_id = "docfinqa:document:shared"

    train = _make_split(
        root=tmp_path,
        split="train",
        document_id=shared_document_id,
        example_id="docfinqa:example:train",
    )
    test = _make_split(
        root=tmp_path,
        split="test",
        document_id=shared_document_id,
        example_id="docfinqa:example:test",
    )

    _write_manifest(
        root=tmp_path,
        splits={
            "train": train,
            "test": test,
        },
    )

    report = validate_docfinqa_output(tmp_path)

    assert len(report.document_overlaps) == 1

    overlap = report.document_overlaps[0]

    assert overlap.count == 1
    assert overlap.document_ids == (shared_document_id,)
    assert {
        overlap.left_split.value,
        overlap.right_split.value,
    } == {
        "train",
        "test",
    }


def test_integrity_rejects_cross_split_example_overlap(
    tmp_path: Path,
) -> None:
    shared_example_id = "docfinqa:example:shared"

    train = _make_split(
        root=tmp_path,
        split="train",
        example_id=shared_example_id,
    )
    test = _make_split(
        root=tmp_path,
        split="test",
        example_id=shared_example_id,
    )

    _write_manifest(
        root=tmp_path,
        splits={
            "train": train,
            "test": test,
        },
    )

    with pytest.raises(
        ValueError,
        match="share 1 example IDs",
    ):
        validate_docfinqa_output(tmp_path)


def test_integrity_rejects_unknown_supporting_element(
    tmp_path: Path,
) -> None:
    train = _make_split(
        root=tmp_path,
        split="train",
        supporting_element_id="unknown-element",
    )
    _write_manifest(
        root=tmp_path,
        splits={"train": train},
    )

    with pytest.raises(
        ValueError,
        match="unknown supporting element",
    ):
        validate_docfinqa_output(tmp_path)


def test_integrity_rejects_checksum_mismatch(
    tmp_path: Path,
) -> None:
    train = _make_split(
        root=tmp_path,
        split="train",
    )
    train["artifacts"]["elements"]["sha256"] = "0" * 64

    _write_manifest(
        root=tmp_path,
        splits={"train": train},
    )

    with pytest.raises(
        ValueError,
        match="elements SHA-256 mismatch",
    ):
        validate_docfinqa_output(tmp_path)


def test_integrity_rejects_invalid_offsets(
    tmp_path: Path,
) -> None:
    train = _make_split(
        root=tmp_path,
        split="train",
        element_end_char=100,
    )
    _write_manifest(
        root=tmp_path,
        splits={"train": train},
    )

    with pytest.raises(
        ValueError,
        match=("source text length does not match character offsets"),
    ):
        validate_docfinqa_output(tmp_path)


def test_integrity_rejects_inconsistent_statistics(
    tmp_path: Path,
) -> None:
    train = _make_split(
        root=tmp_path,
        split="train",
    )
    train["statistics"]["total_records"] = 2
    train["statistics"]["normalized_records"] = 2

    _write_manifest(
        root=tmp_path,
        splits={"train": train},
    )

    with pytest.raises(
        ValueError,
        match=("normalized record count does not match examples"),
    ):
        validate_docfinqa_output(tmp_path)


def test_integrity_rejects_wrong_dataset(
    tmp_path: Path,
) -> None:
    train = _make_split(
        root=tmp_path,
        split="train",
    )
    _write_manifest(
        root=tmp_path,
        splits={"train": train},
    )

    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset"] = "finqa"

    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Manifest dataset must be 'docfinqa'",
    ):
        validate_docfinqa_output(tmp_path)
