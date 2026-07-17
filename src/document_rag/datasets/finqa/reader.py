import json
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from document_rag.datasets.config import DatasetConfig
from document_rag.datasets.finqa.raw_models import FinQARawRecord
from document_rag.datasets.models import DatasetName, DatasetSplit

RECORDS_ADAPTER = TypeAdapter(list[FinQARawRecord])


class FinQARawReader:
    """Read and validate raw FinQA records."""

    def __init__(self, *, source_directory: Path, config: DatasetConfig) -> None:
        if config.name is not DatasetName.FINQA:
            raise ValueError("FinQARawReader requires a FinQA configuration")

        self._source_directory = source_directory.resolve()
        self._config = config

    def read_split(self, split: DatasetSplit) -> list[FinQARawRecord]:
        source_path = self._resolve_source_path(self._config.files.for_split(split))

        with source_path.open(encoding="utf-8") as source_file:
            raw_data = json.load(source_file)

        if not isinstance(raw_data, list):
            raise ValueError(f"Expected a JSON array in {source_path}")

        try:
            return RECORDS_ADAPTER.validate_python(raw_data)
        except ValidationError as error:
            raise ValueError(f"Invalid FinQA data in {source_path}") from error

    def _resolve_source_path(self, relative_path: str) -> Path:
        source_path = (self._source_directory / relative_path).resolve()

        if not source_path.is_relative_to(self._source_directory):
            raise ValueError("Dataset file must be inside the source directory")

        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        return source_path
