from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from time import monotonic
from typing import cast

from document_rag.datasets.config import load_dataset_config
from document_rag.datasets.docfinqa.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
)
from document_rag.datasets.docfinqa.integrity import (
    validate_docfinqa_output,
)
from document_rag.datasets.docfinqa.service import (
    DEFAULT_EVIDENCE_MINIMUM_SCORE,
    DocFinQAProgressCallback,
    DocFinQAProgressEvent,
    DocFinQAProgressStage,
    PreparedDocFinQASplitResult,
    prepare_docfinqa_dataset,
)
from document_rag.datasets.docfinqa.writer import WrittenDocFinQASplit
from document_rag.datasets.finqa import prepare_finqa_dataset
from document_rag.datasets.finqa.writer import WrittenFinQASplit
from document_rag.datasets.models import DatasetName, DatasetSplit
from document_rag.rag.config import DEFAULT_MAX_INPUT_TOKENS, RAGConfig
from document_rag.rag.errors import RAGError
from document_rag.rag.evaluation import run_frozen_rag_evaluation
from document_rag.rag.generation import QwenBaseLoraComparisonGenerator
from document_rag.retrieval.benchmark import run_bm25_benchmark
from document_rag.retrieval.dense_benchmark import run_dense_benchmark
from document_rag.retrieval.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_MODEL_REVISION,
    SentenceTransformerEmbedder,
)
from document_rag.retrieval.evaluation import DEFAULT_K_VALUES
from document_rag.retrieval.hybrid import DEFAULT_CANDIDATE_K, DEFAULT_RRF_K
from document_rag.retrieval.hybrid_benchmark import run_hybrid_benchmark
from document_rag.training import (
    RAGTrainingExportConfig,
    export_financial_qa_training_data,
    export_rag_training_data,
)

_DOCFINQA_PROGRESS_INTERVAL_SECONDS = 5.0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the document-rag command-line interface."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)

    if arguments.command == "data" and arguments.data_command == "prepare":
        return _run_data_prepare(arguments)

    if arguments.command == "data" and arguments.data_command == "validate":
        return _run_data_validate(arguments)

    if arguments.command == "training" and arguments.training_command == "export":
        return _run_training_export(arguments)

    if arguments.command == "training" and arguments.training_command == "export-rag":
        return _run_rag_training_export(arguments)

    if arguments.command == "retrieval" and arguments.retrieval_command == "bm25":
        return _run_retrieval_bm25(arguments)

    if arguments.command == "retrieval" and arguments.retrieval_command == "dense":
        return _run_retrieval_dense(arguments)

    if arguments.command == "retrieval" and arguments.retrieval_command == "hybrid":
        return _run_retrieval_hybrid(arguments)

    if arguments.command == "rag" and arguments.rag_command == "evaluate":
        return _run_rag_evaluate(arguments)

    parser.error("A command is required")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="document-rag",
        description=("Auditable RAG for financial and business documents."),
    )

    commands = parser.add_subparsers(dest="command")

    data_parser = commands.add_parser(
        "data",
        help="Prepare and validate datasets.",
    )
    data_commands = data_parser.add_subparsers(dest="data_command")

    prepare_parser = data_commands.add_parser(
        "prepare",
        help="Prepare a dataset in the normalized project format.",
    )
    prepare_parser.add_argument(
        "--dataset",
        choices=tuple(dataset.value for dataset in DatasetName),
        required=True,
        help="Dataset to prepare.",
    )
    prepare_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the primary dataset TOML configuration.",
    )
    prepare_parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the primary source dataset.",
    )
    prepare_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for normalized artifacts.",
    )
    prepare_parser.add_argument(
        "--split",
        dest="splits",
        action="append",
        choices=tuple(split.value for split in DatasetSplit),
        help=(
            "Split to prepare. May be specified more than once. "
            "All splits are prepared when omitted."
        ),
    )
    prepare_parser.add_argument(
        "--finqa-config",
        type=Path,
        help=("Path to FinQA TOML configuration. Required when preparing DocFinQA."),
    )
    prepare_parser.add_argument(
        "--finqa-source",
        type=Path,
        help=("Path to the FinQA source dataset. Required when preparing DocFinQA."),
    )
    prepare_parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=(f"DocFinQA chunk size in characters. Default: {DEFAULT_CHUNK_SIZE}."),
    )
    prepare_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=(f"DocFinQA chunk overlap in characters. Default: {DEFAULT_CHUNK_OVERLAP}."),
    )
    prepare_parser.add_argument(
        "--evidence-minimum-score",
        type=float,
        default=DEFAULT_EVIDENCE_MINIMUM_SCORE,
        help=(
            f"Minimum DocFinQA evidence matching score. Default: {DEFAULT_EVIDENCE_MINIMUM_SCORE}."
        ),
    )

    validate_parser = data_commands.add_parser(
        "validate",
        help="Validate prepared dataset artifacts.",
    )
    validate_parser.add_argument(
        "--dataset",
        choices=(DatasetName.DOCFINQA.value,),
        required=True,
        help="Prepared dataset to validate.",
    )
    validate_parser.add_argument(
        "--input",
        dest="input_directory",
        type=Path,
        required=True,
        help=("Directory containing manifest.json and normalized artifacts."),
    )

    training_parser = commands.add_parser(
        "training",
        help="Export normalized datasets for model training.",
    )
    training_commands = training_parser.add_subparsers(dest="training_command")

    export_parser = training_commands.add_parser(
        "export",
        help="Export FinQA and DocFinQA as deterministic chat JSONL.",
    )
    export_parser.add_argument(
        "--finqa",
        dest="finqa_directory",
        type=Path,
        required=True,
        help="Directory containing prepared FinQA artifacts.",
    )
    export_parser.add_argument(
        "--docfinqa",
        dest="docfinqa_directory",
        type=Path,
        required=True,
        help="Directory containing prepared DocFinQA artifacts.",
    )
    export_parser.add_argument(
        "--output",
        dest="output_directory",
        type=Path,
        required=True,
        help="Directory for exported training artifacts.",
    )
    export_parser.add_argument(
        "--split",
        dest="splits",
        action="append",
        choices=tuple(split.value for split in DatasetSplit),
        help=(
            "Split to export. May be specified more than once. "
            "All splits are exported when omitted."
        ),
    )

    rag_export_parser = training_commands.add_parser(
        "export-rag",
        help="Export production-shaped RAG chat data for the next adapter.",
    )
    rag_export_parser.add_argument(
        "--finqa",
        dest="finqa_directory",
        type=Path,
        required=True,
        help="Directory containing prepared FinQA artifacts.",
    )
    rag_export_parser.add_argument(
        "--docfinqa",
        dest="docfinqa_directory",
        type=Path,
        required=True,
        help="Directory containing prepared DocFinQA artifacts.",
    )
    rag_export_parser.add_argument(
        "--output",
        dest="output_directory",
        type=Path,
        required=True,
        help="Directory for RAG-aligned JSONL and its manifest.",
    )
    rag_export_parser.add_argument(
        "--split",
        dest="splits",
        action="append",
        choices=tuple(split.value for split in DatasetSplit),
        help="Split to export; repeat as needed. All splits are exported when omitted.",
    )
    rag_export_parser.add_argument(
        "--refusal-ratio",
        type=float,
        default=0.2,
        help="Target fraction of exact insufficient-context refusal examples (default: 0.2).",
    )
    rag_export_parser.add_argument(
        "--oracle-augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Insert missing gold sources for supported generator supervision (default: enabled).",
    )
    rag_export_parser.add_argument(
        "--document-resplit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assign source-report groups to deterministic 80/10/10 splits (default: enabled).",
    )
    rag_export_parser.add_argument(
        "--max-sequence-tokens",
        type=int,
        default=DEFAULT_MAX_INPUT_TOKENS,
        help=(
            "Maximum tokens in the complete templated training sequence, including the target. "
            f"Default: {DEFAULT_MAX_INPUT_TOKENS}."
        ),
    )

    retrieval_parser = commands.add_parser(
        "retrieval",
        help="Run retrieval benchmarks.",
    )
    retrieval_commands = retrieval_parser.add_subparsers(dest="retrieval_command")

    bm25_parser = retrieval_commands.add_parser(
        "bm25",
        help="Evaluate BM25 against normalized source lineage.",
    )
    bm25_parser.add_argument(
        "--finqa",
        dest="finqa_directory",
        type=Path,
        help="Directory containing prepared FinQA artifacts.",
    )
    bm25_parser.add_argument(
        "--docfinqa",
        dest="docfinqa_directory",
        type=Path,
        help="Directory containing prepared DocFinQA artifacts.",
    )
    bm25_parser.add_argument(
        "--output",
        dest="output_directory",
        type=Path,
        required=True,
        help="Directory for retrieval predictions and metrics.",
    )
    bm25_parser.add_argument(
        "--split",
        choices=tuple(split.value for split in DatasetSplit),
        default=DatasetSplit.TEST.value,
        help=f"Prepared split to evaluate. Default: {DatasetSplit.TEST.value}.",
    )
    bm25_parser.add_argument(
        "--top-k",
        dest="top_k_values",
        action="append",
        type=int,
        help=("Rank cutoff to evaluate. May be specified more than once. Defaults to 1, 3, and 5."),
    )

    dense_parser = retrieval_commands.add_parser(
        "dense",
        help="Evaluate exact dense retrieval against the frozen BM25 corpus.",
    )
    dense_parser.add_argument(
        "--finqa",
        dest="finqa_directory",
        type=Path,
        help="Directory containing the same prepared FinQA artifacts as BM25.",
    )
    dense_parser.add_argument(
        "--docfinqa",
        dest="docfinqa_directory",
        type=Path,
        help="Directory containing the same prepared DocFinQA artifacts as BM25.",
    )
    dense_parser.add_argument(
        "--bm25-metrics",
        type=Path,
        required=True,
        help="Frozen BM25 retrieval_metrics.json used for comparison.",
    )
    dense_parser.add_argument(
        "--output",
        dest="output_directory",
        type=Path,
        required=True,
        help="Directory for dense predictions, metrics, and comparison artifacts.",
    )
    dense_parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Sentence Transformers model ID. Default: {DEFAULT_EMBEDDING_MODEL}.",
    )
    dense_parser.add_argument(
        "--model-revision",
        default=DEFAULT_EMBEDDING_MODEL_REVISION,
        help=(
            f"Immutable Hugging Face model revision. Default: {DEFAULT_EMBEDDING_MODEL_REVISION}."
        ),
    )
    dense_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding inference batch size. Default: 32.",
    )
    dense_parser.add_argument(
        "--device",
        default="auto",
        help="Sentence Transformers device, for example auto, cpu, or cuda. Default: auto.",
    )
    dense_parser.add_argument(
        "--cache-directory",
        type=Path,
        help="Optional embedding cache directory. Defaults inside the output directory.",
    )
    dense_parser.add_argument(
        "--split",
        choices=tuple(split.value for split in DatasetSplit),
        default=DatasetSplit.TEST.value,
        help=f"Prepared split to evaluate. Default: {DatasetSplit.TEST.value}.",
    )
    dense_parser.add_argument(
        "--top-k",
        dest="top_k_values",
        action="append",
        type=int,
        help=("Rank cutoff to evaluate. May be specified more than once. Defaults to 1, 3, and 5."),
    )

    hybrid_parser = retrieval_commands.add_parser(
        "hybrid",
        help="Evaluate deterministic BM25 and dense reciprocal-rank fusion.",
    )
    hybrid_parser.add_argument(
        "--finqa",
        dest="finqa_directory",
        type=Path,
        help="Directory containing the same prepared FinQA artifacts as both baselines.",
    )
    hybrid_parser.add_argument(
        "--docfinqa",
        dest="docfinqa_directory",
        type=Path,
        help="Directory containing the same prepared DocFinQA artifacts as both baselines.",
    )
    hybrid_parser.add_argument(
        "--bm25-metrics",
        type=Path,
        required=True,
        help="Frozen BM25 retrieval_metrics.json used for comparison.",
    )
    hybrid_parser.add_argument(
        "--dense-metrics",
        type=Path,
        required=True,
        help="Frozen dense_metrics.json used for comparison.",
    )
    hybrid_parser.add_argument(
        "--output",
        dest="output_directory",
        type=Path,
        required=True,
        help="Directory for hybrid predictions, metrics, and comparison artifacts.",
    )
    hybrid_parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help=f"Reciprocal-rank denominator constant. Default: {DEFAULT_RRF_K}.",
    )
    hybrid_parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_CANDIDATE_K,
        help=f"Candidate depth requested from each component. Default: {DEFAULT_CANDIDATE_K}.",
    )
    hybrid_parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Sentence Transformers model ID. Default: {DEFAULT_EMBEDDING_MODEL}.",
    )
    hybrid_parser.add_argument(
        "--model-revision",
        default=DEFAULT_EMBEDDING_MODEL_REVISION,
        help=(
            f"Immutable Hugging Face model revision. Default: {DEFAULT_EMBEDDING_MODEL_REVISION}."
        ),
    )
    hybrid_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding inference batch size. Default: 32.",
    )
    hybrid_parser.add_argument(
        "--device",
        default="auto",
        help="Sentence Transformers device, for example auto, cpu, or cuda. Default: auto.",
    )
    hybrid_parser.add_argument(
        "--cache-directory",
        type=Path,
        help="Optional dense embedding cache directory. Defaults inside the output directory.",
    )
    hybrid_parser.add_argument(
        "--split",
        choices=tuple(split.value for split in DatasetSplit),
        default=DatasetSplit.TEST.value,
        help=f"Prepared split to evaluate. Default: {DatasetSplit.TEST.value}.",
    )
    hybrid_parser.add_argument(
        "--top-k",
        dest="top_k_values",
        action="append",
        type=int,
        help=("Rank cutoff to evaluate. May be specified more than once. Defaults to 1, 3, and 5."),
    )

    rag_parser = commands.add_parser(
        "rag",
        help="Run end-to-end RAG workflows.",
    )
    rag_commands = rag_parser.add_subparsers(dest="rag_command")
    evaluate_parser = rag_commands.add_parser(
        "evaluate",
        help="Compare the pinned base model and LoRA on frozen RAG contexts.",
    )
    evaluate_parser.add_argument(
        "--suite",
        dest="suite_path",
        type=Path,
        required=True,
        help="Versioned JSON file containing questions and frozen retrieval results.",
    )
    evaluate_parser.add_argument(
        "--output",
        dest="output_directory",
        type=Path,
        required=True,
        help="Directory for paired predictions and aggregate metrics.",
    )
    evaluate_parser.add_argument(
        "--base-model",
        help="Override DOCUMENT_RAG_BASE_MODEL_ID for this evaluation.",
    )
    evaluate_parser.add_argument(
        "--base-revision",
        help="Override DOCUMENT_RAG_BASE_MODEL_REVISION for this evaluation.",
    )
    evaluate_parser.add_argument(
        "--adapter-model",
        help="Override DOCUMENT_RAG_ADAPTER_MODEL_ID for this evaluation.",
    )
    evaluate_parser.add_argument(
        "--adapter-revision",
        help="Override DOCUMENT_RAG_ADAPTER_MODEL_REVISION for this evaluation.",
    )
    evaluate_parser.add_argument(
        "--device-map",
        help="Override DOCUMENT_RAG_GENERATION_DEVICE_MAP, for example auto or cuda.",
    )
    evaluate_parser.add_argument(
        "--max-input-tokens",
        type=int,
        help="Override DOCUMENT_RAG_MAX_INPUT_TOKENS.",
    )
    evaluate_parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Override DOCUMENT_RAG_MAX_NEW_TOKENS.",
    )

    return parser


def _run_data_prepare(arguments: argparse.Namespace) -> int:
    dataset = DatasetName(cast(str, arguments.dataset))
    config_path = cast(Path, arguments.config)
    source_directory = cast(Path, arguments.source)
    output_directory = cast(Path, arguments.output)
    split_values = cast(list[str] | None, arguments.splits)

    splits = _parse_splits(split_values)

    if dataset is DatasetName.FINQA:
        return _run_finqa_prepare(
            config_path=config_path,
            source_directory=source_directory,
            output_directory=output_directory,
            splits=splits,
        )

    if dataset is DatasetName.DOCFINQA:
        return _run_docfinqa_prepare(
            arguments=arguments,
            config_path=config_path,
            source_directory=source_directory,
            output_directory=output_directory,
            splits=splits,
        )

    print(
        f"error: unsupported dataset: {dataset.value}",
        file=sys.stderr,
    )
    return 1


def _run_finqa_prepare(
    *,
    config_path: Path,
    source_directory: Path,
    output_directory: Path,
    splits: tuple[DatasetSplit, ...],
) -> int:
    try:
        config = load_dataset_config(config_path)
        result = prepare_finqa_dataset(
            config=config,
            source_directory=source_directory,
            output_directory=output_directory,
            splits=splits,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Prepared dataset: {DatasetName.FINQA.value}")

    for written_split in result.splits:
        _print_finqa_split(written_split)

    print(f"Manifest: {result.manifest.path}")

    if result.sample_splits:
        print("Prepared deterministic sample:")

        for written_split in result.sample_splits:
            _print_finqa_split(written_split)

        if result.sample_manifest is not None:
            print(f"Sample manifest: {result.sample_manifest.path}")

    for overlap in result.report_overlaps:
        print(
            "Warning: "
            f"{overlap.count} reports are shared between "
            f"{overlap.left_split.value} and "
            f"{overlap.right_split.value}",
            file=sys.stderr,
        )

    return 0


def _run_docfinqa_prepare(
    *,
    arguments: argparse.Namespace,
    config_path: Path,
    source_directory: Path,
    output_directory: Path,
    splits: tuple[DatasetSplit, ...],
) -> int:
    finqa_config_path = cast(Path | None, arguments.finqa_config)
    finqa_source_directory = cast(Path | None, arguments.finqa_source)

    if finqa_config_path is None or finqa_source_directory is None:
        missing_arguments: list[str] = []

        if finqa_config_path is None:
            missing_arguments.append("--finqa-config")

        if finqa_source_directory is None:
            missing_arguments.append("--finqa-source")

        print(
            "error: DocFinQA requires " + " and ".join(missing_arguments),
            file=sys.stderr,
        )
        return 1

    chunk_size = cast(int, arguments.chunk_size)
    chunk_overlap = cast(int, arguments.chunk_overlap)
    evidence_minimum_score = cast(
        float,
        arguments.evidence_minimum_score,
    )

    try:
        docfinqa_config = load_dataset_config(config_path)
        finqa_config = load_dataset_config(finqa_config_path)

        split_names = ", ".join(
            split.value for split in sorted(splits, key=lambda item: item.value)
        )
        print(
            f"Preparing DocFinQA splits: {split_names}",
            file=sys.stderr,
            flush=True,
        )

        result = prepare_docfinqa_dataset(
            docfinqa_source_directory=source_directory,
            finqa_source_directory=finqa_source_directory,
            output_directory=output_directory,
            docfinqa_config=docfinqa_config,
            finqa_config=finqa_config,
            splits=splits,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            evidence_minimum_score=evidence_minimum_score,
            progress_callback=_create_docfinqa_progress_reporter(),
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Prepared dataset: {DatasetName.DOCFINQA.value}")

    for split_result in result.splits:
        _print_docfinqa_split(split_result)

    print(f"Manifest: {result.manifest.path}")

    sample_splits = getattr(result, "sample_splits", ())
    sample_manifest = getattr(result, "sample_manifest", None)

    if sample_splits:
        print("Prepared deterministic sample:")

        for written_split in sample_splits:
            _print_docfinqa_written_split(written_split)

        if sample_manifest is not None:
            print(f"Sample manifest: {sample_manifest.path}")

    for overlap in result.integrity_report.document_overlaps:
        print(
            "Warning: "
            f"{overlap.count} documents are shared between "
            f"{overlap.left_split.value} and "
            f"{overlap.right_split.value}",
            file=sys.stderr,
        )

    return 0


def _create_docfinqa_progress_reporter() -> DocFinQAProgressCallback:
    last_reported_at: dict[DatasetSplit, float] = {}

    def report(event: DocFinQAProgressEvent) -> None:
        stats = event.stats

        if event.stage is DocFinQAProgressStage.STARTED:
            last_reported_at[event.split] = monotonic()
            print(
                f"[{event.split.value}] Starting preparation...",
                file=sys.stderr,
                flush=True,
            )
            return

        if event.stage is DocFinQAProgressStage.COMPLETED:
            print(
                f"[{event.split.value}] Completed: "
                f"{stats.total_records} records processed, "
                f"{stats.normalized_records} normalized, "
                f"{stats.skipped_records} skipped.",
                file=sys.stderr,
                flush=True,
            )
            return

        current_time = monotonic()
        previous_report_time = last_reported_at.get(event.split, current_time)

        if current_time - previous_report_time < _DOCFINQA_PROGRESS_INTERVAL_SECONDS:
            return

        last_reported_at[event.split] = current_time
        print(
            f"[{event.split.value}] {stats.total_records} records processed, "
            f"{stats.normalized_records} normalized, "
            f"{stats.skipped_records} skipped...",
            file=sys.stderr,
            flush=True,
        )

    return report


def _run_data_validate(arguments: argparse.Namespace) -> int:
    dataset = DatasetName(cast(str, arguments.dataset))
    input_directory = cast(Path, arguments.input_directory)

    if dataset is not DatasetName.DOCFINQA:
        print(
            f"error: integrity validation is not implemented for dataset: {dataset.value}",
            file=sys.stderr,
        )
        return 1

    try:
        report = validate_docfinqa_output(input_directory)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Validated dataset: {dataset.value}")

    for snapshot in report.splits:
        print(
            f"  {snapshot.split.value}: "
            f"{len(snapshot.document_ids)} documents, "
            f"{len(snapshot.element_ids)} elements, "
            f"{len(snapshot.example_ids)} examples"
        )

    for overlap in report.document_overlaps:
        print(
            "Warning: "
            f"{overlap.count} documents are shared between "
            f"{overlap.left_split.value} and "
            f"{overlap.right_split.value}",
            file=sys.stderr,
        )

    return 0


def _run_training_export(arguments: argparse.Namespace) -> int:
    finqa_directory = cast(Path, arguments.finqa_directory)
    docfinqa_directory = cast(Path, arguments.docfinqa_directory)
    output_directory = cast(Path, arguments.output_directory)
    split_values = cast(list[str] | None, arguments.splits)

    try:
        result = export_financial_qa_training_data(
            finqa_directory=finqa_directory,
            docfinqa_directory=docfinqa_directory,
            output_directory=output_directory,
            splits=_parse_splits(split_values),
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print("Exported financial QA training data:")

    for artifact in result.artifacts:
        print(f"  {artifact.split.value}: {artifact.record_count} examples -> {artifact.path}")

    print(f"Manifest: {result.manifest_path}")
    return 0


def _run_rag_training_export(arguments: argparse.Namespace) -> int:
    finqa_directory = cast(Path, arguments.finqa_directory)
    docfinqa_directory = cast(Path, arguments.docfinqa_directory)
    output_directory = cast(Path, arguments.output_directory)
    split_values = cast(list[str] | None, arguments.splits)
    refusal_ratio = cast(float, arguments.refusal_ratio)
    oracle_augment = cast(bool, arguments.oracle_augment)
    document_resplit = cast(bool, arguments.document_resplit)
    max_sequence_tokens = cast(int, arguments.max_sequence_tokens)

    try:
        result = export_rag_training_data(
            finqa_directory=finqa_directory,
            docfinqa_directory=docfinqa_directory,
            output_directory=output_directory,
            splits=_parse_splits(split_values),
            rag_config=RAGConfig.from_environment(),
            export_config=RAGTrainingExportConfig(
                refusal_ratio=refusal_ratio,
                oracle_augment=oracle_augment,
                document_resplit=document_resplit,
                max_sequence_tokens=max_sequence_tokens,
            ),
        )
    except (OSError, ValueError, RAGError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print("Exported RAG-aligned adapter training data:")

    for artifact in result.artifacts:
        print(
            f"  {artifact.split.value}: {artifact.record_count} examples "
            f"({artifact.supported_count} supported, {artifact.refusal_count} refusals, "
            f"{artifact.calculation_supervised_count} calculation supervised, "
            f"{artifact.context_trimmed_count} context trimmed, "
            f"{artifact.oracle_augmented_count} oracle augmented, "
            f"{artifact.ambiguous_unit_exclusion_count} ambiguous-unit exclusions, "
            f"{artifact.gold_source_overflow_exclusion_count} gold-source overflow exclusions, "
            f"{artifact.sequence_overflow_exclusion_count} sequence overflow exclusions, "
            f"max {artifact.max_sequence_token_count} tokens) "
            f"-> {artifact.path}"
        )

    print(f"Manifest: {result.manifest_path}")
    return 0


def _run_retrieval_bm25(arguments: argparse.Namespace) -> int:
    finqa_directory = cast(Path | None, arguments.finqa_directory)
    docfinqa_directory = cast(Path | None, arguments.docfinqa_directory)
    output_directory = cast(Path, arguments.output_directory)
    split = DatasetSplit(cast(str, arguments.split))
    top_k_values = cast(list[int] | None, arguments.top_k_values)
    dataset_directories: dict[DatasetName, Path] = {}

    if finqa_directory is not None:
        dataset_directories[DatasetName.FINQA] = finqa_directory

    if docfinqa_directory is not None:
        dataset_directories[DatasetName.DOCFINQA] = docfinqa_directory

    if not dataset_directories:
        print(
            "error: retrieval bm25 requires --finqa and/or --docfinqa",
            file=sys.stderr,
        )
        return 1

    try:
        if top_k_values is None:
            result = run_bm25_benchmark(
                dataset_directories=dataset_directories,
                output_directory=output_directory,
                split=split,
            )
        else:
            result = run_bm25_benchmark(
                dataset_directories=dataset_directories,
                output_directory=output_directory,
                split=split,
                k_values=tuple(top_k_values),
            )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print("Completed BM25 retrieval benchmark:")
    print(f"  {result.metrics.query_count} queries")
    print(f"  {result.document_count} documents")
    print(f"  {result.indexed_chunk_count} indexed chunks")

    for k, value in result.metrics.hit_rate_at_k:
        print(f"  Hit Rate@{k}: {value:.6f}")

    for k, value in result.metrics.recall_at_k:
        print(f"  Recall@{k}: {value:.6f}")

    print(f"  MRR: {result.metrics.mrr:.6f}")
    print(f"Predictions: {result.predictions_path}")
    print(f"Metrics: {result.metrics_path}")
    return 0


def _run_retrieval_dense(arguments: argparse.Namespace) -> int:
    finqa_directory = cast(Path | None, arguments.finqa_directory)
    docfinqa_directory = cast(Path | None, arguments.docfinqa_directory)
    output_directory = cast(Path, arguments.output_directory)
    bm25_metrics_path = cast(Path, arguments.bm25_metrics)
    cache_directory = cast(Path | None, arguments.cache_directory)
    model_id = cast(str, arguments.embedding_model)
    model_revision = cast(str, arguments.model_revision)
    batch_size = cast(int, arguments.batch_size)
    device = cast(str, arguments.device)
    split = DatasetSplit(cast(str, arguments.split))
    top_k_values = cast(list[int] | None, arguments.top_k_values)
    dataset_directories: dict[DatasetName, Path] = {}

    if finqa_directory is not None:
        dataset_directories[DatasetName.FINQA] = finqa_directory

    if docfinqa_directory is not None:
        dataset_directories[DatasetName.DOCFINQA] = docfinqa_directory

    if not dataset_directories:
        print(
            "error: retrieval dense requires --finqa and/or --docfinqa",
            file=sys.stderr,
        )
        return 1

    try:
        embedder = SentenceTransformerEmbedder(
            model_id=model_id,
            model_revision=model_revision,
            device=device,
            show_progress=True,
        )
        result = run_dense_benchmark(
            dataset_directories=dataset_directories,
            output_directory=output_directory,
            bm25_metrics_path=bm25_metrics_path,
            embedder=embedder,
            batch_size=batch_size,
            split=split,
            cache_directory=cache_directory,
            k_values=(DEFAULT_K_VALUES if top_k_values is None else tuple(top_k_values)),
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print("Completed dense retrieval benchmark:")
    print(f"  Model: {embedder.model_id}@{embedder.model_revision}")
    print(f"  Device: {embedder.device}")
    print(f"  Embedding dimension: {embedder.dimension}")
    print(f"  {result.metrics.query_count} queries")
    print(f"  {result.document_count} documents")
    print(f"  {result.indexed_chunk_count} indexed chunks")

    for k, value in result.metrics.hit_rate_at_k:
        print(f"  Hit Rate@{k}: {value:.6f}")

    for k, value in result.metrics.recall_at_k:
        print(f"  Recall@{k}: {value:.6f}")

    print(f"  MRR: {result.metrics.mrr:.6f}")
    cache_status = "reused" if result.embedding_cache_reused else "created"
    print(f"Embedding cache ({cache_status}): {result.embeddings_path}")
    print(f"Predictions: {result.predictions_path}")
    print(f"Metrics: {result.metrics_path}")
    print(f"Comparison: {result.comparison_path}")
    return 0


def _run_retrieval_hybrid(arguments: argparse.Namespace) -> int:
    finqa_directory = cast(Path | None, arguments.finqa_directory)
    docfinqa_directory = cast(Path | None, arguments.docfinqa_directory)
    output_directory = cast(Path, arguments.output_directory)
    bm25_metrics_path = cast(Path, arguments.bm25_metrics)
    dense_metrics_path = cast(Path, arguments.dense_metrics)
    cache_directory = cast(Path | None, arguments.cache_directory)
    model_id = cast(str, arguments.embedding_model)
    model_revision = cast(str, arguments.model_revision)
    rrf_k = cast(int, arguments.rrf_k)
    candidate_k = cast(int, arguments.candidate_k)
    batch_size = cast(int, arguments.batch_size)
    device = cast(str, arguments.device)
    split = DatasetSplit(cast(str, arguments.split))
    top_k_values = cast(list[int] | None, arguments.top_k_values)
    dataset_directories: dict[DatasetName, Path] = {}

    if finqa_directory is not None:
        dataset_directories[DatasetName.FINQA] = finqa_directory

    if docfinqa_directory is not None:
        dataset_directories[DatasetName.DOCFINQA] = docfinqa_directory

    if not dataset_directories:
        print(
            "error: retrieval hybrid requires --finqa and/or --docfinqa",
            file=sys.stderr,
        )
        return 1

    try:
        embedder = SentenceTransformerEmbedder(
            model_id=model_id,
            model_revision=model_revision,
            device=device,
            show_progress=True,
        )
        result = run_hybrid_benchmark(
            dataset_directories=dataset_directories,
            output_directory=output_directory,
            bm25_metrics_path=bm25_metrics_path,
            dense_metrics_path=dense_metrics_path,
            embedder=embedder,
            rrf_k=rrf_k,
            candidate_k=candidate_k,
            batch_size=batch_size,
            split=split,
            cache_directory=cache_directory,
            k_values=(DEFAULT_K_VALUES if top_k_values is None else tuple(top_k_values)),
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print("Completed hybrid RRF retrieval benchmark:")
    print(f"  RRF k: {rrf_k}")
    print(f"  Candidate depth: {candidate_k}")
    print(f"  Model: {embedder.model_id}@{embedder.model_revision}")
    print(f"  Device: {embedder.device}")
    print(f"  Embedding dimension: {embedder.dimension}")
    print(f"  {result.metrics.query_count} queries")
    print(f"  {result.document_count} documents")
    print(f"  {result.indexed_chunk_count} indexed chunks")

    for k, value in result.metrics.hit_rate_at_k:
        print(f"  Hit Rate@{k}: {value:.6f}")

    for k, value in result.metrics.recall_at_k:
        print(f"  Recall@{k}: {value:.6f}")

    print(f"  MRR: {result.metrics.mrr:.6f}")
    cache_status = "reused" if result.embedding_cache_reused else "created"
    print(f"Embedding cache ({cache_status}): {result.embeddings_path}")
    print(f"Predictions: {result.predictions_path}")
    print(f"Metrics: {result.metrics_path}")
    print(f"Comparison: {result.comparison_path}")
    return 0


def _run_rag_evaluate(arguments: argparse.Namespace) -> int:
    suite_path = cast(Path, arguments.suite_path)
    output_directory = cast(Path, arguments.output_directory)

    try:
        config = _rag_evaluation_config(arguments)
        result = run_frozen_rag_evaluation(
            suite_path=suite_path,
            output_directory=output_directory,
            generator=QwenBaseLoraComparisonGenerator(config),
            config=config,
        )
    except (OSError, RAGError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print("Completed frozen-context RAG evaluation:")
    print(f"  Suite: {result.suite.name} ({len(result.suite.cases)} cases)")
    print(
        "  Base overall: "
        f"{result.base_metrics.overall.correct}/{result.base_metrics.overall.count} "
        f"({_format_optional_rate(result.base_metrics.overall.accuracy)})"
    )
    print(
        "  Adapter overall: "
        f"{result.adapter_metrics.overall.correct}/{result.adapter_metrics.overall.count} "
        f"({_format_optional_rate(result.adapter_metrics.overall.accuracy)})"
    )
    print(f"Predictions: {result.predictions_path}")
    print(f"Metrics: {result.metrics_path}")
    return 0


def _rag_evaluation_config(arguments: argparse.Namespace) -> RAGConfig:
    config = RAGConfig.from_environment()
    base_model = cast(str | None, arguments.base_model)
    base_revision = cast(str | None, arguments.base_revision)
    adapter_model = cast(str | None, arguments.adapter_model)
    adapter_revision = cast(str | None, arguments.adapter_revision)
    device_map = cast(str | None, arguments.device_map)
    max_input_tokens = cast(int | None, arguments.max_input_tokens)
    max_new_tokens = cast(int | None, arguments.max_new_tokens)
    return replace(
        config,
        base_model_id=config.base_model_id if base_model is None else base_model,
        base_model_revision=(
            config.base_model_revision if base_revision is None else base_revision
        ),
        adapter_model_id=config.adapter_model_id if adapter_model is None else adapter_model,
        adapter_model_revision=(
            config.adapter_model_revision if adapter_revision is None else adapter_revision
        ),
        generation_device_map=(config.generation_device_map if device_map is None else device_map),
        max_input_tokens=(
            config.max_input_tokens if max_input_tokens is None else max_input_tokens
        ),
        max_new_tokens=(config.max_new_tokens if max_new_tokens is None else max_new_tokens),
    )


def _format_optional_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def _parse_splits(
    split_values: list[str] | None,
) -> tuple[DatasetSplit, ...]:
    if split_values:
        return tuple(DatasetSplit(value) for value in split_values)

    return tuple(DatasetSplit)


def _print_finqa_split(written_split: WrittenFinQASplit) -> None:
    print(
        f"  {written_split.split.value}: "
        f"{written_split.documents.record_count} documents, "
        f"{written_split.elements.record_count} elements, "
        f"{written_split.examples.record_count} examples"
    )


def _print_docfinqa_split(
    split_result: PreparedDocFinQASplitResult,
) -> None:
    written_split = split_result.written_split
    stats = split_result.stats

    _print_docfinqa_written_split(written_split)

    if stats.skipped_records:
        print(
            "    skipped: "
            f"{stats.skipped_records} total "
            f"({stats.skipped_ambiguous} ambiguous, "
            f"{stats.skipped_answer_mismatch} answer mismatch, "
            f"{stats.skipped_evidence_incomplete} "
            "incomplete evidence, "
            f"{stats.skipped_duplicate} duplicates)"
        )


def _print_docfinqa_written_split(
    written_split: WrittenDocFinQASplit,
) -> None:
    print(
        f"  {written_split.split.value}: "
        f"{written_split.documents.record_count} documents, "
        f"{written_split.elements.record_count} elements, "
        f"{written_split.examples.record_count} examples"
    )
