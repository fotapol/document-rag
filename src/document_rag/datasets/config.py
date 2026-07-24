import tomllib
from pathlib import Path
from typing import Annotated

from pydantic import Field

from document_rag.datasets.models import DatasetName, DatasetSplit
from document_rag.domain.base import BaseDomainModel
from document_rag.domain.types import NonEmptyString

GitCommitSha = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


class DatasetFiles(BaseDomainModel):
    """Source files corresponding to normalized dataset splits."""

    train: NonEmptyString
    validation: NonEmptyString
    test: NonEmptyString

    def for_split(self, split: DatasetSplit) -> str:
        match split:
            case DatasetSplit.TRAIN:
                return self.train
            case DatasetSplit.VALIDATION:
                return self.validation
            case DatasetSplit.TEST:
                return self.test


class DatasetConfig(BaseDomainModel):
    """Versioned configuration of an external dataset."""

    name: DatasetName
    schema_version: NonEmptyString
    source_url: NonEmptyString
    source_revision: GitCommitSha
    files: DatasetFiles


def load_dataset_config(path: Path) -> DatasetConfig:
    """Load and validate a dataset TOML configuration."""

    with path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    return DatasetConfig.model_validate(raw_config)
