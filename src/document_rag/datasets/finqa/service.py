from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from document_rag.datasets.config import DatasetConfig
from document_rag.datasets.finqa.integrity import (
    ReportOverlap,
    SplitIntegritySnapshot,
    build_split_integrity_snapshot,
    validate_cross_split_integrity,
)
from document_rag.datasets.finqa.manifest import (
    WrittenManifest,
    write_finqa_manifest,
)
from document_rag.datasets.finqa.preparer import (
    PreparedFinQASplit,
    prepare_finqa_split,
)
from document_rag.datasets.finqa.reader import FinQARawReader
from document_rag.datasets.finqa.sample import create_finqa_sample
from document_rag.datasets.finqa.writer import (
    WrittenFinQASplit,
    write_finqa_split,
)
from document_rag.datasets.models import DatasetName, DatasetSplit

_SAMPLE_EXAMPLE_COUNTS = {
    DatasetSplit.TRAIN: 128,
    DatasetSplit.VALIDATION: 32,
}


@dataclass(frozen=True, slots=True)
class FinQAPreparationResult:
    """Artifacts and integrity information produced by FinQA preparation."""

    splits: tuple[WrittenFinQASplit, ...]
    manifest: WrittenManifest
    sample_splits: tuple[WrittenFinQASplit, ...]
    sample_manifest: WrittenManifest | None
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

    prepared_splits: dict[DatasetSplit, PreparedFinQASplit] = {}
    written_splits: list[WrittenFinQASplit] = []
    integrity_snapshots: list[SplitIntegritySnapshot] = []
    report_overlaps: list[ReportOverlap] = []

    for split in selected_splits:
        prepared_split = prepare_finqa_split(
            reader,
            split=split,
        )
        prepared_splits[split] = prepared_split

        integrity_snapshot = build_split_integrity_snapshot(prepared_split)

        report_overlaps.extend(
            validate_cross_split_integrity(
                integrity_snapshot,
                previous_snapshots=integrity_snapshots,
            )
        )

        integrity_snapshots.append(integrity_snapshot)

        written_splits.append(
            write_finqa_split(
                prepared_split,
                output_directory=output_directory,
            )
        )

    ordered_splits = _order_written_splits(written_splits)

    manifest = write_finqa_manifest(
        config=config,
        written_splits=ordered_splits,
        output_directory=output_directory,
    )

    sample_splits = _write_samples(
        prepared_splits=prepared_splits,
        output_directory=output_directory / "sample",
    )

    sample_manifest = (
        write_finqa_manifest(
            config=config,
            written_splits=sample_splits,
            output_directory=output_directory / "sample",
        )
        if sample_splits
        else None
    )

    return FinQAPreparationResult(
        splits=ordered_splits,
        manifest=manifest,
        sample_splits=sample_splits,
        sample_manifest=sample_manifest,
        report_overlaps=tuple(report_overlaps),
    )


def _write_samples(
    *,
    prepared_splits: dict[DatasetSplit, PreparedFinQASplit],
    output_directory: Path,
) -> tuple[WrittenFinQASplit, ...]:
    written_samples: list[WrittenFinQASplit] = []

    for split, requested_count in _SAMPLE_EXAMPLE_COUNTS.items():
        prepared_split = prepared_splits.get(split)

        if prepared_split is None:
            continue

        if not prepared_split.examples:
            raise ValueError(f"Cannot create a sample from empty {split.value!r} split")

        sample = create_finqa_sample(
            prepared_split,
            example_count=min(
                requested_count,
                len(prepared_split.examples),
            ),
        )

        written_samples.append(
            write_finqa_split(
                sample,
                output_directory=output_directory,
            )
        )

    return _order_written_splits(written_samples)


def _order_written_splits(
    written_splits: Sequence[WrittenFinQASplit],
) -> tuple[WrittenFinQASplit, ...]:
    return tuple(
        sorted(
            written_splits,
            key=lambda item: item.split.value,
        )
    )
