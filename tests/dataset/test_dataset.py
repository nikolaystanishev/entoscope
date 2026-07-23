import pytest

from intex.dataset.config import DatasetConfig
from intex.dataset.dataset import AbstractDataset, Row
from intex.dataset.factory import DatasetFactory

ROWS = [
    {"prompt": "Paris is located in", "label": 1},
    {"prompt": "Rome is located in", "label": 0},
]


@pytest.fixture
def dataset(list_dataset_class) -> AbstractDataset:
    return DatasetFactory.create(DatasetConfig(type="list", rows=ROWS))


def test_prepare_returns_self_and_loads_once(dataset):
    assert dataset.prepare() is dataset
    dataset.prepare()
    assert dataset.load_calls == 1


def test_rows_are_available_after_prepare(dataset):
    dataset.prepare()
    assert dataset.rows == ROWS
    assert len(dataset) == 2
    assert list(dataset) == ROWS
    assert dataset[1] == {"prompt": "Rome is located in", "label": 0}


def test_rows_before_prepare_raises(dataset):
    with pytest.raises(AssertionError):
        _ = dataset.rows
    with pytest.raises(AssertionError):
        len(dataset)


def test_implementation_reads_its_settings_off_the_dynamic_config(list_dataset_class):
    class LimitedDataset(list_dataset_class):
        def load(self) -> list[Row]:
            return super().load()[: self.config.get("limit")]

    DatasetFactory.register("limited")(LimitedDataset)
    dataset = DatasetFactory.create(DatasetConfig(type="limited", rows=ROWS, limit=1)).prepare()
    assert dataset.rows == ROWS[:1]


def test_abstract_dataset_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AbstractDataset(DatasetConfig(type="list"))
