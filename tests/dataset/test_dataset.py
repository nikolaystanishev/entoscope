import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import pytest

from entoscope.dataset.config import DatasetConfig
from entoscope.dataset.dataset import AbstractDataset, Row
from entoscope.dataset.factory import DatasetFactory
from entoscope.operation.config import ConfigFactory

from .conftest import ROWS, dataset_config


def _load(path: str, **properties) -> AbstractDataset:
    return DatasetFactory.create(dataset_config(path, **properties)).prepare()


def test_a_prepared_dataset_exposes_the_rows_of_its_file(jsonl_dataset_class, prompts_file):
    dataset = _load(prompts_file())

    assert dataset.rows == ROWS
    assert len(dataset) == len(ROWS)
    assert list(dataset) == ROWS
    assert dataset[1] == ROWS[1]


def test_the_rows_are_loaded_once(jsonl_dataset_class, prompts_file):
    path = prompts_file()
    dataset = DatasetFactory.create(dataset_config(path))

    assert dataset.prepare() is dataset
    with open(path, "w", encoding="utf-8") as f:
        f.write('{"prompt": "something else", "label": 1}\n')

    # A second prepare keeps what it already loaded instead of reading the file again.
    assert dataset.prepare().rows == ROWS


def test_rows_are_unavailable_before_the_dataset_is_prepared(jsonl_dataset_class, prompts_file):
    dataset = DatasetFactory.create(dataset_config(prompts_file()))

    with pytest.raises(AssertionError):
        _ = dataset.rows
    with pytest.raises(AssertionError):
        len(dataset)


def test_an_implementation_reads_its_settings_off_the_dynamic_config(jsonl_dataset_class, prompts_file):
    dataset = _load(prompts_file(), limit=2)

    assert dataset.rows == ROWS[:2]


def test_a_property_the_config_does_not_have_raises(jsonl_dataset_class, tmp_path):
    dataset = DatasetFactory.create(DatasetConfig(type="jsonl"))

    with pytest.raises(AttributeError, match="Dataset config 'jsonl' has no property 'path'"):
        dataset.prepare()


def test_a_class_without_a_load_cannot_be_registered():
    class Incomplete(AbstractDataset):
        pass

    with pytest.raises(TypeError, match="Incomplete cannot be registered as dataset type 'incomplete'.*load"):
        DatasetFactory.register("incomplete")(Incomplete)

    assert DatasetFactory.registered() == []


# --------------------------------------------------------------------------------------------------
# Configuration — 'type' is declared, everything else is a dynamic property of the implementation
# --------------------------------------------------------------------------------------------------
def test_a_yaml_config_becomes_type_plus_dynamic_properties(jsonl_dataset_class, config_from_yaml, prompts_file):
    path = prompts_file()
    raw = {"type": "jsonl", "path": path, "limit": 2, "columns": {"prompt": "text"}, "tags": ["train", "clean"]}

    config = config_from_yaml(DatasetConfig, "my_data", raw)

    assert config.type == "jsonl"
    assert config.config_id == "my_data"
    assert config.properties == {"path": path, "limit": 2, "columns": {"prompt": "text"}, "tags": ["train", "clean"]}
    # Nothing knows what a dataset's keys mean, so nested shapes arrive as plain dict/list.
    assert config.columns == {"prompt": "text"}
    assert config.tags == ["train", "clean"]
    assert config.get("limit") == 2
    assert config.get("missing", 5) == 5
    assert "limit" in config and "missing" not in config

    assert DatasetFactory.create(config).prepare().rows == ROWS[:2]


def test_a_yaml_config_and_a_code_built_one_load_the_same_rows(jsonl_dataset_class, config_from_yaml, prompts_file):
    path = prompts_file()
    from_file = config_from_yaml(DatasetConfig, "my_data", {"type": "jsonl", "path": path})

    assert DatasetFactory.create(from_file).prepare().rows == _load(path).rows


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"path": "prompts.jsonl"}, "Missing required keys for DatasetConfig.*type"),
        ("jsonl", "Expected a mapping for DatasetConfig"),
    ],
)
def test_a_malformed_dataset_config_is_rejected_when_it_is_loaded(config_from_yaml, raw, message):
    with pytest.raises(ValueError, match=message):
        config_from_yaml(DatasetConfig, "broken", raw)


def test_an_already_built_config_passes_through(jsonl_dataset_class, prompts_file):
    config = dataset_config(prompts_file())

    assert DatasetFactory.create(config).config is config


# --------------------------------------------------------------------------------------------------
# A config subclass — its own fields are typed, everything else stays a dynamic property
# --------------------------------------------------------------------------------------------------
class Split(str, Enum):
    TRAIN = "train"
    TEST = "test"


@dataclass(kw_only=True)
class SplitDatasetConfig(DatasetConfig):
    split: Split = Split.TRAIN
    columns: dict[str, Any] = field(default_factory=dict)


class SplitDataset(AbstractDataset):
    """Reads the rows of the file named after its split, e.g. ``prompts.test.jsonl``."""

    def load(self) -> list[Row]:
        with open(f"{self.config.path}.{self.config.split.value}.jsonl", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


def test_a_subclass_field_is_typed_while_the_extras_stay_dynamic(config_from_yaml, prompts_file, tmp_path):
    DatasetFactory.register("split")(SplitDataset)
    prompts_file(ROWS[:2], name="prompts.test.jsonl")
    raw = {"type": "split", "split": "test", "path": str(tmp_path / "prompts")}

    config = config_from_yaml(SplitDatasetConfig, "my_data", raw)

    assert isinstance(config, SplitDatasetConfig)
    assert config.split is Split.TEST
    assert config.properties == {"path": str(tmp_path / "prompts")}
    assert DatasetFactory.create(config).prepare().rows == ROWS[:2]


def test_a_subclass_field_falls_back_to_its_default(config_from_yaml):
    config = config_from_yaml(SplitDatasetConfig, "my_data", {"type": "split"})

    assert config.split is Split.TRAIN
    assert config.columns == {}


def test_a_declared_field_that_takes_anything_keeps_its_values_as_they_are(config_from_yaml):
    # 'columns' is typed as a mapping of anything, so its values arrive as the YAML wrote them.
    columns = {"prompt": "text", "labels": [1, 0], "meta": {"source": "hand"}}

    config = config_from_yaml(SplitDatasetConfig, "my_data", {"type": "split", "columns": columns})

    assert config.columns == columns
    assert config.properties == {}  # a declared field is not a dynamic property


def test_a_config_file_may_name_environment_variables(monkeypatch, jsonl_dataset_class, config_from_yaml, prompts_file):
    # read_yaml resolves ${VAR} while loading: on its own it becomes the value (as an int when the
    # value is a number), and inside a longer string it is substituted in place.
    path = prompts_file()
    monkeypatch.setenv("ENTOSCOPE_TEST_DIR", str(Path(path).parent))
    monkeypatch.setenv("ENTOSCOPE_TEST_LIMIT", "2")
    raw = {"type": "jsonl", "path": "${ENTOSCOPE_TEST_DIR}/prompts.jsonl", "limit": "${ENTOSCOPE_TEST_LIMIT}"}

    config = config_from_yaml(DatasetConfig, "my_data", raw, file_name="env.yaml")

    assert config.path == path
    assert config.limit == 2
    assert DatasetFactory.create(config).prepare().rows == ROWS[:2]


def test_a_value_tagged_for_the_environment_without_naming_one_is_left_alone(
    jsonl_dataset_class, prompts_file, tmp_path
):
    # The !ENV tag can also be written by hand, on a value that names no variable at all.
    path = prompts_file()
    config_yaml = tmp_path / "tagged.yaml"
    config_yaml.write_text(f'my_data:\n  type: jsonl\n  path: !ENV "{path}"\n')

    config = ConfigFactory[DatasetConfig](str(config_yaml)).create("my_data")

    assert config.path == path
    assert DatasetFactory.create(config).prepare().rows == ROWS


def test_an_environment_variable_that_is_not_set_stays_as_its_name(config_from_yaml):
    config = config_from_yaml(DatasetConfig, "my_data", {"type": "jsonl", "path": "${ENTOSCOPE_UNSET}"}, "unset.yaml")

    assert config.path == "ENTOSCOPE_UNSET"
