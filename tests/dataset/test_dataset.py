import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import pytest
import yaml

from entoscope.dataset.config import DatasetConfig
from entoscope.dataset.dataset import AbstractDataset, Row
from entoscope.dataset.factory import DatasetFactory
from entoscope.operation.config import ConfigFactory
from entoscope.util.data import from_dict, read_yaml, to_dict

from .conftest import ROWS, JsonlDatasetConfig, dataset_config


def _load(path: str, **settings) -> AbstractDataset:
    return DatasetFactory.create(dataset_config(path, **settings)).prepare()


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


def test_an_implementation_reads_its_settings_off_its_own_config(jsonl_dataset_class, prompts_file):
    dataset = _load(prompts_file(), limit=2)

    assert dataset.rows == ROWS[:2]


def test_a_class_without_a_load_cannot_be_registered():
    class Incomplete(AbstractDataset):
        pass

    with pytest.raises(TypeError, match="Incomplete cannot be registered as dataset type 'incomplete'.*load"):
        DatasetFactory.register("incomplete")(Incomplete)

    assert DatasetFactory.registered() == []


# --------------------------------------------------------------------------------------------------
# Configuration - an entry is read into the config class its type's dataset names
# --------------------------------------------------------------------------------------------------
def test_a_yaml_config_is_read_into_the_config_its_dataset_names(jsonl_dataset_class, config_from_yaml, prompts_file):
    path = prompts_file()

    config = config_from_yaml(DatasetConfig, "my_data", {"type": "jsonl", "path": path, "limit": 2})

    assert config == JsonlDatasetConfig(type="jsonl", config_id="my_data", path=path, limit=2)
    assert DatasetFactory.create(config).prepare().rows == ROWS[:2]


def test_a_yaml_config_and_a_code_built_one_load_the_same_rows(jsonl_dataset_class, config_from_yaml, prompts_file):
    path = prompts_file()
    from_file = config_from_yaml(DatasetConfig, "my_data", {"type": "jsonl", "path": path})

    assert DatasetFactory.create(from_file).prepare().rows == _load(path).rows


def test_a_saved_config_reads_back_into_the_same_config(jsonl_dataset_class, prompts_file):
    config = dataset_config(prompts_file(), limit=2)

    assert from_dict(DatasetConfig, to_dict(config)) == config


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"path": "prompts.jsonl"}, r"Missing required keys for DatasetConfig: \['type'\]"),
        ({"type": "csv", "path": "prompts.csv"}, "Unknown dataset type: csv"),
        ({"type": "jsonl"}, r"Missing required keys for JsonlDatasetConfig: \['path'\]"),
        ({"type": "jsonl", "path": "prompts.jsonl", "columns": ["text"]}, r"Unknown keys for JsonlDatasetConfig"),
        (["jsonl"], "Expected a dataset type or a mapping for DatasetConfig"),
    ],
)
def test_a_malformed_dataset_config_is_rejected_when_it_is_loaded(jsonl_dataset_class, config_from_yaml, raw, message):
    with pytest.raises(ValueError, match=message):
        config_from_yaml(DatasetConfig, "broken", raw)


def test_an_already_built_config_passes_through(jsonl_dataset_class, prompts_file):
    config = dataset_config(prompts_file())

    assert DatasetFactory.create(config).config is config


# --------------------------------------------------------------------------------------------------
# A config of typed settings - fields are read and checked like any other config's
# --------------------------------------------------------------------------------------------------
class Split(StrEnum):
    TRAIN = "train"
    TEST = "test"


@dataclass(kw_only=True)
class SplitDatasetConfig(DatasetConfig):
    path: str = "prompts"
    split: Split = Split.TRAIN
    columns: dict[str, Any] = field(default_factory=dict)


class SplitDataset(AbstractDataset[SplitDatasetConfig]):
    """Reads the rows of the file named after its split, e.g. ``prompts.test.jsonl``."""

    def load(self) -> list[Row]:
        with open(f"{self.config.path}.{self.config.split.value}.jsonl", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


@pytest.fixture
def split_dataset_class():
    return DatasetFactory.register("split")(SplitDataset)


def test_a_field_is_read_into_its_type(split_dataset_class, config_from_yaml, prompts_file, tmp_path):
    prompts_file(ROWS[:2], name="prompts.test.jsonl")
    raw = {"type": "split", "split": "test", "path": str(tmp_path / "prompts")}

    config = config_from_yaml(DatasetConfig, "my_data", raw)

    assert isinstance(config, SplitDatasetConfig)
    assert config.split is Split.TEST
    assert DatasetFactory.create(config).prepare().rows == ROWS[:2]


def test_a_dataset_named_by_its_type_alone_takes_its_defaults(split_dataset_class, config_from_yaml):
    config = config_from_yaml(DatasetConfig, "my_data", "split")

    assert config == SplitDatasetConfig(type="split", config_id="my_data")


def test_a_declared_field_that_takes_anything_keeps_its_values_as_they_are(split_dataset_class, config_from_yaml):
    # 'columns' is typed as a mapping of anything, so its values arrive as the YAML wrote them.
    columns = {"prompt": "text", "labels": [1, 0], "meta": {"source": "hand"}}

    config = config_from_yaml(DatasetConfig, "my_data", {"type": "split", "columns": columns})

    assert config.columns == columns


# --------------------------------------------------------------------------------------------------
# Environment variables in a config file
# --------------------------------------------------------------------------------------------------
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


def test_an_environment_variable_that_is_not_set_stays_as_its_name(jsonl_dataset_class, config_from_yaml):
    config = config_from_yaml(DatasetConfig, "my_data", {"type": "jsonl", "path": "${ENTOSCOPE_UNSET}"}, "unset.yaml")

    assert config.path == "ENTOSCOPE_UNSET"


def test_reading_a_config_file_leaves_other_yaml_loading_alone(monkeypatch, tmp_path):
    monkeypatch.setenv("ENTOSCOPE_TEST_DIR", "/data")
    config_yaml = tmp_path / "env.yaml"
    config_yaml.write_text("path: ${ENTOSCOPE_TEST_DIR}\n")
    resolvers = {key: list(value) for key, value in yaml.FullLoader.yaml_implicit_resolvers.items()}

    assert read_yaml(str(config_yaml)) == read_yaml(str(config_yaml)) == {"path": "/data"}

    assert yaml.FullLoader.yaml_implicit_resolvers == resolvers
    assert yaml.full_load("path: ${ENTOSCOPE_TEST_DIR}") == {"path": "${ENTOSCOPE_TEST_DIR}"}
