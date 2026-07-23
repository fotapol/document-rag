"""Application service for preparing DocFinQA dataset artifacts."""

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from document_rag.datasets.config import DatasetConfig
from document_rag.datasets.docfinqa.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DocFinQAChunker,
)
from document_rag.datasets.docfinqa.evidence import (
    DocFinQAEvidenceSelector,
)
from document_rag.datasets.docfinqa.manifest import (
    DocFinQASplitManifestInput,
    WrittenDocFinQAManifest,
    write_docfinqa_manifest,
)
from document_rag.datasets.docfinqa.normalizer import DocFinQANormalizer
from document_rag.datasets.docfinqa.preparer import (
    DocFinQAPreparationStats,
    DocFinQASplitPreparer,
)
from document_rag.datasets.docfinqa.raw_models import DocFinQARawRecord
from document_rag.datasets.docfinqa.reader import DocFinQARawReader
from document_rag.datasets.docfinqa.sample import (
    DEFAULT_SAMPLE_DOCUMENTS_PER_SPLIT,
    create_docfinqa_sample,
)
from document_rag.datasets.docfinqa.writer import (
    WrittenDocFinQASplit,
    write_docfinqa_split,
)
from document_rag.datasets.finqa.reader import FinQARawReader
from document_rag.datasets.models import DatasetName, DatasetSplit

DEFAULT_EVIDENCE_MINIMUM_SCORE = 0.6

_DEFAULT_SPLITS = (
    DatasetSplit.TRAIN,
    DatasetSplit.VALIDATION,
    DatasetSplit.TEST,
)


@dataclass(frozen=True, slots=True)
class PreparedDocFinQASplitResult:
    """Written artifacts and statistics for one prepared split."""

    written_split: WrittenDocFinQASplit
    stats: DocFinQAPreparationStats


@dataclass(frozen=True, slots=True)
class DocFinQAPreparationResult:
    """Complete result of preparing DocFinQA."""

    splits: tuple[PreparedDocFinQASplitResult, ...]
    manifest: WrittenDocFinQAManifest
    sample_splits: tuple[WrittenDocFinQASplit, ...]
    sample_manifest: WrittenDocFinQAManifest


class DocFinQAProgressStage(StrEnum):
    """Stage represented by a DocFinQA progress event."""

    STARTED = "started"
    PROCESSING = "processing"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class DocFinQAProgressEvent:
    """Current state of one DocFinQA split preparation."""

    split: DatasetSplit
    stage: DocFinQAProgressStage
    stats: DocFinQAPreparationStats


type DocFinQAProgressCallback = Callable[[DocFinQAProgressEvent], None]


def prepare_docfinqa_dataset(
    *,
    docfinqa_source_directory: Path,
    finqa_source_directory: Path,
    output_directory: Path,
    docfinqa_config: DatasetConfig,
    finqa_config: DatasetConfig,
    splits: Sequence[DatasetSplit] = _DEFAULT_SPLITS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    evidence_minimum_score: float = DEFAULT_EVIDENCE_MINIMUM_SCORE,
    sample_documents_per_split: int = (DEFAULT_SAMPLE_DOCUMENTS_PER_SPLIT),
    progress_callback: DocFinQAProgressCallback | None = None,
) -> DocFinQAPreparationResult:
    """Prepare full DocFinQA artifacts and a development subset."""

    if docfinqa_config.name is not DatasetName.DOCFINQA:
        raise ValueError("DocFinQA preparation requires a DocFinQA configuration")

    if finqa_config.name is not DatasetName.FINQA:
        raise ValueError("DocFinQA preparation requires a FinQA configuration")

    if sample_documents_per_split <= 0:
        raise ValueError("Sample documents per split must be positive")

    selected_splits = _validate_splits(splits)

    docfinqa_reader = DocFinQARawReader(
        source_directory=docfinqa_source_directory,
        config=docfinqa_config,
    )
    finqa_reader = FinQARawReader(
        source_directory=finqa_source_directory,
        config=finqa_config,
    )

    prepared_results: list[PreparedDocFinQASplitResult] = []
    manifest_inputs: list[DocFinQASplitManifestInput] = []

    for split in selected_splits:
        finqa_records = finqa_reader.read_split(split)

        normalizer = DocFinQANormalizer(
            chunker=DocFinQAChunker(
                chunk_size=chunk_size,
                overlap=chunk_overlap,
            ),
            evidence_selector=DocFinQAEvidenceSelector(
                minimum_score=evidence_minimum_score,
            ),
        )

        preparer = DocFinQASplitPreparer(
            split=split,
            finqa_records=finqa_records,
            normalizer=normalizer,
        )

        _report_progress(
            progress_callback,
            split=split,
            stage=DocFinQAProgressStage.STARTED,
            stats=preparer.stats,
        )

        raw_records = _iter_with_progress(
            docfinqa_reader.iter_split(split),
            split=split,
            preparer=preparer,
            progress_callback=progress_callback,
        )

        written_split = write_docfinqa_split(
            output_directory=output_directory,
            split=split,
            items=preparer.iter_prepare(raw_records),
        )

        stats = preparer.stats

        _report_progress(
            progress_callback,
            split=split,
            stage=DocFinQAProgressStage.COMPLETED,
            stats=stats,
        )

        prepared_result = PreparedDocFinQASplitResult(
            written_split=written_split,
            stats=stats,
        )
        prepared_results.append(prepared_result)

        manifest_inputs.append(
            DocFinQASplitManifestInput(
                written_split=written_split,
                stats=stats,
            )
        )

    manifest = write_docfinqa_manifest(
        output_directory=output_directory,
        config=docfinqa_config,
        split_inputs=manifest_inputs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        evidence_minimum_score=evidence_minimum_score,
    )

    sample_result = create_docfinqa_sample(
        input_directory=output_directory,
        output_directory=output_directory / "sample",
        config=docfinqa_config,
        splits=selected_splits,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        evidence_minimum_score=evidence_minimum_score,
        documents_per_split=sample_documents_per_split,
    )

    return DocFinQAPreparationResult(
        splits=tuple(prepared_results),
        manifest=manifest,
        sample_splits=sample_result.splits,
        sample_manifest=sample_result.manifest,
    )


def _iter_with_progress(
    raw_records: Iterable[DocFinQARawRecord],
    *,
    split: DatasetSplit,
    preparer: DocFinQASplitPreparer,
    progress_callback: DocFinQAProgressCallback | None,
) -> Iterator[DocFinQARawRecord]:
    for raw_record in raw_records:
        yield raw_record

        _report_progress(
            progress_callback,
            split=split,
            stage=DocFinQAProgressStage.PROCESSING,
            stats=preparer.stats,
        )


def _report_progress(
    progress_callback: DocFinQAProgressCallback | None,
    *,
    split: DatasetSplit,
    stage: DocFinQAProgressStage,
    stats: DocFinQAPreparationStats,
) -> None:
    if progress_callback is None:
        return

    progress_callback(
        DocFinQAProgressEvent(
            split=split,
            stage=stage,
            stats=stats,
        )
    )


def _validate_splits(
    splits: Sequence[DatasetSplit],
) -> tuple[DatasetSplit, ...]:
    selected_splits = tuple(splits)

    if not selected_splits:
        raise ValueError("At least one DocFinQA split must be selected")

    if len(set(selected_splits)) != len(selected_splits):
        raise ValueError("DocFinQA split selection contains duplicates")

    return tuple(
        sorted(
            selected_splits,
            key=lambda split: split.value,
        )
    )
