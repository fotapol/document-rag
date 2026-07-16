from pathlib import Path

from document_rag.datasets import DatasetName, DatasetSplit, load_dataset_config


def test_load_dataset_config(tmp_path: Path) -> None:
    config_path = tmp_path / "finqa.toml"
    config_path.write_text(
        """
name = "finqa"
schema_version = "1"
source_url = "https://example.com/finqa.git"
source_revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

[files]
train = "dataset/train.json"
validation = "dataset/dev.json"
test = "dataset/test.json"
""".strip(),
        encoding="utf-8",
    )

    config = load_dataset_config(config_path)

    assert config.name is DatasetName.FINQA
    assert config.files.for_split(DatasetSplit.VALIDATION) == "dataset/dev.json"
