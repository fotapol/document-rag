import json
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel

from document_rag.datasets.finqa.preparer import PreparedFinQASplit
from document_rag.datasets.models import DatasetSplit


@dataclass(frozen=True, slots=True)
class WrittenArtifact:
    """Metadata describing one generated dataset artifact."""

    path: Path
    record_count: int
    checksum_sha256: str


@dataclass(frozen=True, slots=True)
class WrittenFinQASplit:
    """Artifacts generated for one normalized FinQA split."""

    split: DatasetSplit
    documents: WrittenArtifact
    elements: WrittenArtifact
    examples: WrittenArtifact


def write_finqa_split(
    prepared_split: PreparedFinQASplit,
    *,
    output_directory: Path,
) -> WrittenFinQASplit:
    """Write a normalized FinQA split as deterministic JSONL files."""

    split_directory = output_directory / prepared_split.split.value

    return WrittenFinQASplit(
        split=prepared_split.split,
        documents=_write_jsonl(
            split_directory / "documents.jsonl",
            prepared_split.documents,
        ),
        elements=_write_jsonl(
            split_directory / "elements.jsonl",
            prepared_split.elements,
        ),
        examples=_write_jsonl(
            split_directory / "examples.jsonl",
            prepared_split.examples,
        ),
    )


def _write_jsonl[ModelT: BaseModel](
    path: Path,
    records: Iterable[ModelT],
) -> WrittenArtifact:
    path.parent.mkdir(parents=True, exist_ok=True)

    materialized_records = tuple(records)
    temporary_path = path.with_name(f".{path.name}.tmp")

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as output_file:
            for record in materialized_records:
                serialized = json.dumps(
                    record.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                output_file.write(serialized)
                output_file.write("\n")

        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return WrittenArtifact(
        path=path,
        record_count=len(materialized_records),
        checksum_sha256=sha256(path.read_bytes()).hexdigest(),
    )
