import pytest

from entoscope.dataset.config import DatasetConfig
from entoscope.dataset.dataset import AbstractDataset, Row
from entoscope.dataset.factory import DatasetFactory

from .conftest import ROWS, JsonlDataset, dataset_config


def test_a_registered_type_resolves_to_its_implementation(jsonl_dataset_class, prompts_file):
    assert jsonl_dataset_class is JsonlDataset
    assert DatasetFactory.get("jsonl") is JsonlDataset
    assert DatasetFactory.registered() == ["jsonl"]

    config = dataset_config(prompts_file())
    dataset = DatasetFactory.create(config)
    assert isinstance(dataset, JsonlDataset)
    assert dataset.config is config
    assert dataset.prepare().rows == ROWS


def test_an_unknown_type_lists_what_is_registered(jsonl_dataset_class):
    with pytest.raises(ValueError, match=r"Unknown dataset type: csv\. Registered dataset types: \['jsonl'\]"):
        DatasetFactory.get("csv")

    with pytest.raises(ValueError, match="Unknown dataset type: csv"):
        DatasetFactory.create(DatasetConfig(type="csv"))


def test_a_name_cannot_be_taken_over_by_another_class(jsonl_dataset_class):
    class OtherDataset(AbstractDataset):
        def load(self) -> list[Row]:
            return []

    with pytest.raises(ValueError, match="Dataset type 'jsonl' is already registered to JsonlDataset"):
        DatasetFactory.register("jsonl")(OtherDataset)

    assert DatasetFactory.get("jsonl") is JsonlDataset


CELL = """
from entoscope.dataset.dataset import AbstractDataset


class CellDataset(AbstractDataset):
    def load(self):
        return []
"""


def test_defining_a_class_again_replaces_it():
    # Re-running a notebook cell runs its class statement again, building a new class under the same name.
    first, second = {"__name__": "notebook"}, {"__name__": "notebook"}
    exec(CELL, first)
    exec(CELL, second)

    DatasetFactory.register("cell")(first["CellDataset"])
    DatasetFactory.register("cell")(second["CellDataset"])

    assert DatasetFactory.get("cell") is second["CellDataset"]


def test_classes_one_function_builds_are_two_implementations():
    def dataset_of(rows: list[Row]) -> type[AbstractDataset]:
        class BuiltDataset(AbstractDataset):
            def load(self) -> list[Row]:
                return rows

        return BuiltDataset

    empty = DatasetFactory.register("built")(dataset_of([]))

    with pytest.raises(ValueError, match="Dataset type 'built' is already registered to BuiltDataset"):
        DatasetFactory.register("built")(dataset_of(ROWS))
    assert DatasetFactory.get("built") is empty


def test_registering_the_same_class_again_changes_nothing(jsonl_dataset_class):
    DatasetFactory.register("jsonl")(JsonlDataset)

    assert DatasetFactory.get("jsonl") is JsonlDataset
    assert DatasetFactory.registered() == ["jsonl"]


def test_one_class_can_answer_to_several_names(jsonl_dataset_class, prompts_file):
    DatasetFactory.register("lines")(JsonlDataset)

    assert DatasetFactory.registered() == ["jsonl", "lines"]
    dataset = DatasetFactory.create(DatasetConfig(type="lines", path=prompts_file()))
    assert isinstance(dataset, JsonlDataset)
    assert dataset.prepare().rows == ROWS
