from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from document_rag.datasets.config import load_dataset_config
from document_rag.datasets.finqa import prepare_finqa_dataset
from document_rag.datasets.models import DatasetName, DatasetSplit


def main(argv: Sequence[str] | None = None) -> int:
    """Run the document-rag command-line interface."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)

    if arguments.command == "data" and arguments.data_command == "prepare":
        return _run_data_prepare(arguments)

    parser.error("A command is required")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="document-rag",
        description="Auditable RAG for financial and business documents.",
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
        choices=(DatasetName.FINQA.value,),
        required=True,
        help="Dataset to prepare.",
    )
    prepare_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the dataset TOML configuration.",
    )
    prepare_parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the cloned source dataset.",
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

    return parser


def _run_data_prepare(arguments: argparse.Namespace) -> int:
    dataset = cast(str, arguments.dataset)
    config_path = cast(Path, arguments.config)
    source_directory = cast(Path, arguments.source)
    output_directory = cast(Path, arguments.output)
    split_values = cast(list[str] | None, arguments.splits)

    if dataset != DatasetName.FINQA.value:
        print(
            f"error: unsupported dataset: {dataset}",
            file=sys.stderr,
        )
        return 1

    splits = (
        tuple(DatasetSplit(value) for value in split_values)
        if split_values
        else tuple(DatasetSplit)
    )

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

    print(f"Prepared dataset: {dataset}")

    for written_split in result.splits:
        print(
            f"  {written_split.split.value}: "
            f"{written_split.documents.record_count} documents, "
            f"{written_split.elements.record_count} elements, "
            f"{written_split.examples.record_count} examples"
        )

    print(f"Manifest: {result.manifest.path}")

    for overlap in result.report_overlaps:
        print(
            "Warning: "
            f"{overlap.count} reports are shared between "
            f"{overlap.left_split.value} and "
            f"{overlap.right_split.value}",
            file=sys.stderr,
        )

    return 0
