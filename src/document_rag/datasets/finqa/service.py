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
from document_rag.datasets.integrity import (
    ReportOverlap,
    SplitIntegritySnapshot,
    build_split_integrity_snapshot,
    validate_cross_split_integrity,
)
from document_rag.datasets.models import DatasetName, DatasetSplit


@dataclass(frozen=True, slots=True)
class FinQAPreparationResult:
    """Artifacts and integrity information produced by FinQA preparation."""

    splits: tuple[WrittenFinQASplit, ...]
    manifest: WrittenManifest
    report_overlaps: tuple[ReportOverlap, ...]


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

    selected_splits = tuple(splits)

    if not selected_splits:
        raise ValueError("At least one dataset split must be selected")

    if len(selected_splits) != len(set(selected_splits)):
        raise ValueError("Dataset splits must be unique")

    reader = FinQARawReader(
        source_directory=source_directory,
        config=config,
    )

    written_splits: list[WrittenFinQASplit] = []
    integrity_snapshots: list[SplitIntegritySnapshot] = []
    report_overlaps: list[ReportOverlap] = []

    for split in selected_splits:
        prepared_split = prepare_finqa_split(
            reader,
            split=split,
        )

        integrity_snapshot = build_split_integrity_snapshot(prepared_split)

        report_overlaps.extend(
            validate_cross_split_integrity(
                integrity_snapshot,
                previous_snapshots=integrity_snapshots,
            )
        )

        integrity_snapshots.append(integrity_snapshot)

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
        report_overlaps=tuple(report_overlaps),
    )
