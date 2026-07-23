from copy import deepcopy
from dataclasses import dataclass
from enum import Enum

import pytest
import yaml

from intex.dataset.config import DatasetConfig
from intex.operation.config import ConfigFactory
from intex.util.data import from_dict

RAW = {
    "type": "jsonl",
    "path": "data/prompts.jsonl",
    "limit": 10,
    "columns": {"prompt": "text", "label": "is_true"},
    "tags": ["train", "clean"],
}


def test_declared_type_and_dynamic_properties():
    config = from_dict(DatasetConfig, RAW)
    assert config.type == "jsonl"
    assert config.properties == {
        "path": "data/prompts.jsonl",
        "limit": 10,
        "columns": {"prompt": "text", "label": "is_true"},
        "tags": ["train", "clean"],
    }


def test_yaml_shape_equals_code_built_config():
    assert from_dict(DatasetConfig, RAW) == DatasetConfig(
        type="jsonl",
        path="data/prompts.jsonl",
        limit=10,
        columns={"prompt": "text", "label": "is_true"},
        tags=["train", "clean"],
    )


def test_dynamic_properties_are_not_coerced():
    # Nothing knows what a dataset's keys mean, so nested shapes arrive as plain dict/list.
    config = from_dict(DatasetConfig, RAW)
    assert config.columns == {"prompt": "text", "label": "is_true"}
    assert config.tags == ["train", "clean"]


def test_properties_are_read_as_attributes():
    config = from_dict(DatasetConfig, RAW)
    assert config.path == "data/prompts.jsonl"
    assert config.get("limit") == 10
    assert config.get("missing", 5) == 5
    assert "limit" in config
    assert "missing" not in config


def test_unknown_property_raises_attribute_error():
    config = from_dict(DatasetConfig, RAW)
    with pytest.raises(AttributeError, match="Dataset config 'jsonl' has no property 'split'"):
        _ = config.split


def test_missing_type_raises():
    with pytest.raises(ValueError, match="Missing required keys for DatasetConfig.*type"):
        from_dict(DatasetConfig, {"path": "data/prompts.jsonl"})


def test_non_mapping_raises():
    with pytest.raises(ValueError, match="Expected a mapping for DatasetConfig"):
        from_dict(DatasetConfig, "jsonl")


def test_already_typed_value_passes_through():
    config = DatasetConfig(type="jsonl", path="data/prompts.jsonl")
    assert from_dict(DatasetConfig, config) is config


def test_input_is_not_mutated():
    snapshot = deepcopy(RAW)
    from_dict(DatasetConfig, RAW)
    assert snapshot == RAW


def test_config_factory_stamps_config_id(tmp_path):
    config_yaml = tmp_path / "configs.yaml"
    config_yaml.write_text(yaml.safe_dump({"my_data": RAW}))
    factory = ConfigFactory[DatasetConfig](str(config_yaml))
    config = factory.create("my_data")
    assert isinstance(config, DatasetConfig)
    assert config.config_id == "my_data"
    assert config.path == "data/prompts.jsonl"

    with pytest.raises(KeyError):
        factory.create("missing")


class Split(str, Enum):
    TRAIN = "train"
    TEST = "test"


@dataclass(kw_only=True)
class SplitDatasetConfig(DatasetConfig):
    split: Split = Split.TRAIN


def test_subclass_fields_are_typed_and_extras_stay_dynamic():
    config = from_dict(SplitDatasetConfig, {"type": "jsonl", "split": "test", "path": "data/prompts.jsonl"})
    assert isinstance(config, SplitDatasetConfig)
    assert config.split is Split.TEST
    assert config.properties == {"path": "data/prompts.jsonl"}
    assert config.path == "data/prompts.jsonl"


def test_subclass_field_default_applies():
    config = from_dict(SplitDatasetConfig, {"type": "jsonl"})
    assert config.split is Split.TRAIN
    assert config.properties == {}
