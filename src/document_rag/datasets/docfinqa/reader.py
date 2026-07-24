from collections.abc import Iterator
from pathlib import Path

import ijson  # type: ignore[import-untyped]
from ijson.common import JSONError  # type: ignore[import-untyped]
from pydantic import ValidationError

from document_rag.datasets.config import DatasetConfig
from document_rag.datasets.docfinqa.raw_models import DocFinQARawRecord
from document_rag.datasets.models import DatasetName, DatasetSplit


class DocFinQARawReader:
    """Stream and validate records from local DocFinQA JSON files."""

    def __init__(
        self,
        *,
        source_directory: Path,
        config: DatasetConfig,
    ) -> None:
        if config.name is not DatasetName.DOCFINQA:
            raise ValueError("DocFinQARawReader requires a DocFinQA configuration")

        self._source_directory = source_directory.resolve()
        self._config = config

    def iter_split(
        self,
        split: DatasetSplit,
    ) -> Iterator[DocFinQARawRecord]:
        """Yield validated records without loading the full split."""

        relative_path = self._config.files.for_split(split)
        source_path = self._resolve_source_path(relative_path)
        record_found = False

        try:
            with source_path.open("rb") as source_file:
                raw_records = ijson.items(source_file, "item")

                for index, raw_record in enumerate(raw_records):
                    record_found = True

                    try:
                        yield DocFinQARawRecord.model_validate(raw_record)
                    except ValidationError as error:
                        raise ValueError(
                            f"Invalid DocFinQA record at index {index} in {source_path}"
                        ) from error
        except JSONError as error:
            raise ValueError(f"Invalid JSON structure in {source_path}") from error

        if not record_found:
            raise ValueError(f"DocFinQA split is empty: {source_path}")

    def _resolve_source_path(self, relative_path: str) -> Path:
        source_path = (self._source_directory / relative_path).resolve()

        if not source_path.is_relative_to(self._source_directory):
            raise ValueError("Dataset file must be inside the source directory")

        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        return source_path
