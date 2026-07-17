import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from document_rag.datasets.config import DatasetConfig
from document_rag.datasets.finqa.writer import WrittenArtifact, WrittenFinQASplit
from document_rag.datasets.models import DatasetName, DatasetSplit
from document_rag.domain.base import BaseDomainModel
from document_rag.domain.types import NonEmptyString


class ArtifactManifest(BaseDomainModel):
    """Metadata for one generated dataset artifact."""

    path: NonEmptyString
    record_count: int = Field(ge=0)
    checksum_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class SplitManifest(BaseDomainModel):
    """Generated artifacts belonging to one dataset split."""

    split: DatasetSplit
    documents: ArtifactManifest
    elements: ArtifactManifest
    examples: ArtifactManifest


class FinQAManifest(BaseDomainModel):
    """Reproducibility metadata for a prepared FinQA dataset."""

    dataset: DatasetName
    schema_version: NonEmptyString
    source_url: NonEmptyString
    source_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    splits: tuple[SplitManifest, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.dataset is not DatasetName.FINQA:
            raise ValueError("FinQA manifest requires the FinQA dataset")

        split_names = [split.split for split in self.splits]

        if len(split_names) != len(set(split_names)):
            raise ValueError("Manifest splits must be unique")

        return self


@dataclass(frozen=True, slots=True)
class WrittenManifest:
    """Metadata for a written manifest file."""

    path: Path
    checksum_sha256: str


def write_finqa_manifest(
    *,
    config: DatasetConfig,
    written_splits: Sequence[WrittenFinQASplit],
    output_directory: Path,
) -> WrittenManifest:
    """Create and write a deterministic FinQA manifest."""

    if config.name is not DatasetName.FINQA:
        raise ValueError("FinQA manifest requires a FinQA configuration")

    output_root = output_directory.resolve()

    split_manifests = tuple(
        _build_split_manifest(
            written_split,
            output_root=output_root,
        )
        for written_split in sorted(
            written_splits,
            key=lambda item: item.split.value,
        )
    )

    manifest = FinQAManifest(
        dataset=config.name,
        schema_version=config.schema_version,
        source_url=config.source_url,
        source_revision=config.source_revision,
        splits=split_manifests,
    )

    manifest_path = output_root / "manifest.json"
    temporary_path = manifest_path.with_name(".manifest.json.tmp")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        serialized = json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )

        temporary_path.write_text(
            f"{serialized}\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary_path.replace(manifest_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return WrittenManifest(
        path=manifest_path,
        checksum_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
    )


def _build_split_manifest(
    written_split: WrittenFinQASplit,
    *,
    output_root: Path,
) -> SplitManifest:
    return SplitManifest(
        split=written_split.split,
        documents=_build_artifact_manifest(
            written_split.documents,
            output_root=output_root,
        ),
        elements=_build_artifact_manifest(
            written_split.elements,
            output_root=output_root,
        ),
        examples=_build_artifact_manifest(
            written_split.examples,
            output_root=output_root,
        ),
    )


def _build_artifact_manifest(
    artifact: WrittenArtifact,
    *,
    output_root: Path,
) -> ArtifactManifest:
    artifact_path = artifact.path.resolve()

    try:
        relative_path = artifact_path.relative_to(output_root)
    except ValueError as error:
        raise ValueError("Artifact must be inside the output directory") from error

    return ArtifactManifest(
        path=relative_path.as_posix(),
        record_count=artifact.record_count,
        checksum_sha256=artifact.checksum_sha256,
    )
