import json

import pytest

from intex.dataset.config import DatasetConfig
from intex.dataset.dataset import AbstractDataset, Row
from intex.dataset.factory import DatasetFactory

ROWS = [
    {"prompt": "The movie was absolutely wonderful", "label": 1},
    {"prompt": "I loved every minute of it", "label": 1},
    {"prompt": "The movie was dreadful and boring", "label": 0},
    {"prompt": "I hated every minute of it", "label": 0},
]


@pytest.fixture(autouse=True)
def registry():
    """The registry is class state; snapshot it so registrations don't leak between tests."""
    snapshot = dict(DatasetFactory._registry)
    DatasetFactory._registry.clear()
    yield DatasetFactory._registry
    DatasetFactory._registry.clear()
    DatasetFactory._registry.update(snapshot)


class JsonlDataset(AbstractDataset):
    """Reads its rows from the JSONL file the config names, keeping at most ``limit`` of them."""

    def load(self) -> list[Row]:
        with open(self.config.path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return rows[: self.config.get("limit")]


@pytest.fixture
def jsonl_dataset_class(registry):
    return DatasetFactory.register("jsonl")(JsonlDataset)


@pytest.fixture
def prompts_file(tmp_path):
    """Writes rows to a JSONL file and returns its path."""

    def write(rows=ROWS, name: str = "prompts.jsonl") -> str:
        path = tmp_path / name
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        return str(path)

    return write


def dataset_config(path: str, **properties) -> DatasetConfig:
    return DatasetConfig(type="jsonl", path=path, **properties)
