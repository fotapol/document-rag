"""Deterministic manifest generation for prepared DocFinQA datasets."""

import json
import os
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from document_rag.datasets.config import DatasetConfig
from document_rag.datasets.docfinqa.preparer import (
    DocFinQAPreparationStats,
)
from document_rag.datasets.docfinqa.writer import (
    WrittenDocFinQAArtifact,
    WrittenDocFinQASplit,
)
from document_rag.datasets.models import DatasetName

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class DocFinQASplitManifestInput:
    """Written artifacts and preparation statistics for one split."""

    written_split: WrittenDocFinQASplit
    stats: DocFinQAPreparationStats


@dataclass(frozen=True, slots=True)
class WrittenDocFinQAManifest:
    """Metadata for the generated manifest file."""

    path: Path
    sha256: str


def write_docfinqa_manifest(
    *,
    output_directory: Path,
    config: DatasetConfig,
    finqa_config: DatasetConfig,
    split_inputs: Iterable[DocFinQASplitManifestInput],
    chunk_size: int,
    chunk_overlap: int,
    evidence_minimum_score: float,
    sample_documents_per_split: int | None = None,
) -> WrittenDocFinQAManifest:
    """Write a deterministic manifest for prepared DocFinQA data."""

    if config.name is not DatasetName.DOCFINQA:
        raise ValueError("DocFinQA manifest requires a DocFinQA configuration")

    if finqa_config.name is not DatasetName.FINQA:
        raise ValueError("DocFinQA manifest requires a FinQA configuration")

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be non-negative and smaller than chunk size")

    if not 0.0 < evidence_minimum_score <= 1.0:
        raise ValueError("Evidence minimum score must be greater than 0 and no greater than 1")

    if sample_documents_per_split is not None and sample_documents_per_split <= 0:
        raise ValueError("Sample documents per split must be positive")

    resolved_output_directory = output_directory.resolve()
    resolved_output_directory.mkdir(parents=True, exist_ok=True)

    ordered_inputs = sorted(
        split_inputs,
        key=lambda item: item.written_split.split.value,
    )

    if not ordered_inputs:
        raise ValueError("At least one DocFinQA split is required")

    split_names = [item.written_split.split for item in ordered_inputs]

    if len(set(split_names)) != len(split_names):
        raise ValueError("DocFinQA manifest contains duplicate splits")

    split_payloads: dict[str, Any] = {}

    for split_input in ordered_inputs:
        written_split = split_input.written_split
        stats = split_input.stats

        _validate_split(
            output_directory=resolved_output_directory,
            written_split=written_split,
            stats=stats,
        )

        split_payloads[written_split.split.value] = {
            "artifacts": {
                "documents": _serialize_artifact(
                    artifact=written_split.documents,
                    output_directory=resolved_output_directory,
                ),
                "elements": _serialize_artifact(
                    artifact=written_split.elements,
                    output_directory=resolved_output_directory,
                ),
                "examples": _serialize_artifact(
                    artifact=written_split.examples,
                    output_directory=resolved_output_directory,
                ),
            },
            "statistics": _serialize_stats(stats),
        }

    payload: dict[str, Any] = {
        "dataset": DatasetName.DOCFINQA.value,
        "schema_version": config.schema_version,
        "sources": {
            DatasetName.DOCFINQA.value: {
                "revision": config.source_revision,
                "url": config.source_url,
            },
            DatasetName.FINQA.value: {
                "revision": finqa_config.source_revision,
                "url": finqa_config.source_url,
            },
        },
        "preprocessing": {
            "chunk_overlap": chunk_overlap,
            "chunk_size": chunk_size,
            "chunk_stride": chunk_size - chunk_overlap,
            "evidence_minimum_score": evidence_minimum_score,
        },
        "splits": split_payloads,
    }

    if sample_documents_per_split is not None:
        payload["sample"] = {
            "kind": "development_subset",
            "requested_documents_per_split": sample_documents_per_split,
            "selection": "sha256(document_id)",
        }

    manifest_bytes = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    manifest_path = resolved_output_directory / "manifest.json"

    _write_atomically(
        path=manifest_path,
        content=manifest_bytes,
    )

    return WrittenDocFinQAManifest(
        path=manifest_path,
        sha256=sha256(manifest_bytes).hexdigest(),
    )


def _validate_split(
    *,
    output_directory: Path,
    written_split: WrittenDocFinQASplit,
    stats: DocFinQAPreparationStats,
) -> None:
    artifacts = (
        written_split.documents,
        written_split.elements,
        written_split.examples,
    )

    for artifact in artifacts:
        _validate_artifact(
            artifact=artifact,
            output_directory=output_directory,
        )

    if stats.total_records != (stats.normalized_records + stats.skipped_records):
        raise ValueError("DocFinQA preparation statistics do not add up")

    if stats.normalized_records != written_split.examples.record_count:
        raise ValueError("Normalized record count does not match written examples")

    if stats.unique_documents != written_split.documents.record_count:
        raise ValueError("Unique document count does not match written documents")

    linked_records = stats.exact_links + stats.equivalent_links
    accepted_or_evidence_skipped = (
        stats.normalized_records + stats.skipped_evidence_incomplete + stats.skipped_duplicate
    )

    if linked_records != accepted_or_evidence_skipped:
        raise ValueError("DocFinQA linkage statistics do not add up")


def _validate_artifact(
    *,
    artifact: WrittenDocFinQAArtifact,
    output_directory: Path,
) -> None:
    artifact_path = artifact.path.resolve()

    try:
        artifact_path.relative_to(output_directory)
    except ValueError as error:
        raise ValueError("DocFinQA artifact must be inside the output directory") from error

    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)

    if artifact.record_count <= 0:
        raise ValueError("DocFinQA artifact record count must be positive")

    if not _SHA256_PATTERN.fullmatch(artifact.sha256):
        raise ValueError("DocFinQA artifact has an invalid SHA-256 checksum")


def _serialize_artifact(
    *,
    artifact: WrittenDocFinQAArtifact,
    output_directory: Path,
) -> dict[str, Any]:
    relative_path = artifact.path.resolve().relative_to(output_directory)

    return {
        "path": relative_path.as_posix(),
        "record_count": artifact.record_count,
        "sha256": artifact.sha256,
    }


def _serialize_stats(
    stats: DocFinQAPreparationStats,
) -> dict[str, int]:
    return {
        "equivalent_links": stats.equivalent_links,
        "exact_links": stats.exact_links,
        "normalized_records": stats.normalized_records,
        "skipped_ambiguous": stats.skipped_ambiguous,
        "skipped_answer_mismatch": stats.skipped_answer_mismatch,
        "skipped_duplicate": stats.skipped_duplicate,
        "skipped_evidence_incomplete": (stats.skipped_evidence_incomplete),
        "skipped_records": stats.skipped_records,
        "total_records": stats.total_records,
        "unique_documents": stats.unique_documents,
        "unmatched_evidence_facts": stats.unmatched_evidence_facts,
    }


def _write_atomically(
    *,
    path: Path,
    content: bytes,
) -> None:
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        temporary_file.write(content)
        temporary_file.flush()

    try:
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
