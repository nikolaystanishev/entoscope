import pytest

from intex.dataset.config import DatasetConfig
from intex.dataset.dataset import AbstractDataset, Row
from intex.dataset.factory import DatasetFactory


@pytest.fixture(autouse=True)
def registry():
    """The registry is class state; snapshot it so registrations don't leak between tests."""
    snapshot = dict(DatasetFactory._registry)
    DatasetFactory._registry.clear()
    yield DatasetFactory._registry
    DatasetFactory._registry.clear()
    DatasetFactory._registry.update(snapshot)


class ListDataset(AbstractDataset):
    """Rows come from the config's dynamic 'rows' property; 'load_calls' counts the loads."""

    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.load_calls = 0

    def load(self) -> list[Row]:
        self.load_calls += 1
        return list(self.config.get("rows", []))


@pytest.fixture
def list_dataset_class(registry):
    return DatasetFactory.register("list")(ListDataset)
