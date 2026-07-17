from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from document_rag.datasets.config import DatasetConfig
from document_rag.datasets.finqa.manifest import (
    WrittenManifest,
    write_finqa_manifest,
)
from document_rag.datasets.finqa.preparer import prepare_finqa_split
from document_rag.datasets.finqa.reader import FinQARawReader
from document_rag.datasets.finqa.writer import (
    WrittenFinQASplit,
    write_finqa_split,
)
from document_rag.datasets.models import DatasetName, DatasetSplit


@dataclass(frozen=True, slots=True)
class FinQAPreparationResult:
    """Artifacts produced by a complete FinQA preparation run."""

    splits: tuple[WrittenFinQASplit, ...]
    manifest: WrittenManifest


def prepare_finqa_dataset(
    *,
    config: DatasetConfig,
    source_directory: Path,
    output_directory: Path,
    splits: Sequence[DatasetSplit] = tuple(DatasetSplit),
) -> FinQAPreparationResult:
    """Read, normalize, validate, and write selected FinQA splits."""

    if config.name is not DatasetName.FINQA:
        raise ValueError("FinQA preparation requires a FinQA configuration")

    normalized_splits = tuple(splits)

    if not normalized_splits:
        raise ValueError("At least one dataset split must be selected")

    if len(normalized_splits) != len(set(normalized_splits)):
        raise ValueError("Dataset splits must be unique")

    reader = FinQARawReader(
        source_directory=source_directory,
        config=config,
    )

    written_splits: list[WrittenFinQASplit] = []

    for split in normalized_splits:
        prepared_split = prepare_finqa_split(
            reader,
            split=split,
        )

        written_split = write_finqa_split(
            prepared_split,
            output_directory=output_directory,
        )

        written_splits.append(written_split)

    ordered_splits = tuple(
        sorted(
            written_splits,
            key=lambda item: item.split.value,
        )
    )

    manifest = write_finqa_manifest(
        config=config,
        written_splits=ordered_splits,
        output_directory=output_directory,
    )

    return FinQAPreparationResult(
        splits=ordered_splits,
        manifest=manifest,
    )
