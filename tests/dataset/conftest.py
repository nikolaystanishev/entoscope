import json
from dataclasses import dataclass

import pytest

from entoscope.dataset.config import DatasetConfig
from entoscope.dataset.dataset import AbstractDataset, Row
from entoscope.dataset.factory import DatasetFactory

ROWS = [
    {"prompt": "The movie was absolutely wonderful", "label": 1},
    {"prompt": "I loved every minute of it", "label": 1},
    {"prompt": "The movie was dreadful and boring", "label": 0},
    {"prompt": "I hated every minute of it", "label": 0},
]


@pytest.fixture(autouse=True)
def _registries(registries):
    """Every dataset test registers its own types, so each one's registrations are undone after it."""


@dataclass(kw_only=True)
class JsonlDatasetConfig(DatasetConfig):
    path: str
    limit: int | None = None


class JsonlDataset(AbstractDataset[JsonlDatasetConfig]):
    """Reads its rows from the JSONL file the config names, keeping at most ``limit`` of them."""

    def load(self) -> list[Row]:
        with open(self.config.path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return rows[: self.config.limit]


@pytest.fixture
def jsonl_dataset_class():
    return DatasetFactory.register("jsonl")(JsonlDataset)


@pytest.fixture
def prompts_file(tmp_path):
    """Writes rows to a JSONL file and returns its path."""

    def write(rows=ROWS, name: str = "prompts.jsonl") -> str:
        path = tmp_path / name
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        return str(path)

    return write


def dataset_config(path: str, **settings) -> JsonlDatasetConfig:
    return JsonlDatasetConfig(type="jsonl", path=path, **settings)
