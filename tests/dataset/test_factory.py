import pytest

from intex.dataset.config import DatasetConfig
from intex.dataset.dataset import AbstractDataset, Row
from intex.dataset.factory import DatasetFactory

from .conftest import ListDataset


def test_register_returns_the_class(list_dataset_class):
    assert list_dataset_class is ListDataset


def test_get_resolves_registered_type(list_dataset_class):
    assert DatasetFactory.get("list") is ListDataset
    assert DatasetFactory.registered() == ["list"]


def test_create_passes_the_config_through(list_dataset_class):
    config = DatasetConfig(type="list", rows=[{"prompt": "Paris is located in"}])
    dataset = DatasetFactory.create(config)
    assert isinstance(dataset, ListDataset)
    assert dataset.config is config


def test_unknown_type_lists_registered_types(list_dataset_class):
    with pytest.raises(ValueError, match=r"Unknown dataset type: csv\. Registered types: \['list'\]"):
        DatasetFactory.get("csv")

    with pytest.raises(ValueError, match="Unknown dataset type: csv"):
        DatasetFactory.create(DatasetConfig(type="csv"))


def test_duplicate_name_with_another_class_raises(list_dataset_class):
    class OtherDataset(AbstractDataset):
        def load(self) -> list[Row]:
            return []

    with pytest.raises(ValueError, match="Dataset type 'list' is already registered to ListDataset"):
        DatasetFactory.register("list")(OtherDataset)

    assert DatasetFactory.get("list") is ListDataset


def test_registering_the_same_class_again_is_a_noop(list_dataset_class):
    DatasetFactory.register("list")(ListDataset)
    assert DatasetFactory.get("list") is ListDataset
    assert DatasetFactory.registered() == ["list"]


def test_the_same_class_can_be_registered_under_several_names(list_dataset_class):
    DatasetFactory.register("list_copy")(ListDataset)
    assert DatasetFactory.registered() == ["list", "list_copy"]
    assert isinstance(DatasetFactory.create(DatasetConfig(type="list_copy")), ListDataset)
