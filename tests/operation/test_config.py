from copy import deepcopy

import pytest
import yaml

from intex.model.config import Filter, Group, Incremental, ModelLayerLocation, ModelLocation
from intex.operation.config import ConfigFactory, from_dict
from intex.patching.config import (
    PatchingConfig,
    PatchingElementConfig,
    PatchingMetric,
    PatchingType,
    SinglePatchingConfig,
)

ELEMENT = {
    "activation_name": "resid.post",
    "location": ["layer"],
    "filter": [{"layer": -1}],
    "type": "replace",
}


def element_config(**overrides) -> PatchingElementConfig:
    defaults = dict(
        activation_name=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelLocation.LAYER],
        filter=Filter(inclusion_terms=[{"layer": -1}]),
        type=PatchingType.REPLACE,
    )
    return PatchingElementConfig(**{**defaults, **overrides})


def test_yaml_shape_equals_code_built_config():
    config = from_dict(PatchingConfig, {"metric": "logit_diff", "patch": [{"source": ELEMENT, "target": ELEMENT}]})
    element = element_config()
    assert config == PatchingConfig(
        metric=PatchingMetric.LOGIT_DIFF, patch=[SinglePatchingConfig(source=element, target=element)]
    )


def test_enum_coercion_by_value():
    element = from_dict(PatchingElementConfig, ELEMENT)
    assert element.activation_name is ModelLayerLocation.RESIDUAL_POST
    assert element.location == [ModelLocation.LAYER]
    assert element.type is PatchingType.REPLACE


def test_defaults_apply_for_omitted_keys():
    element = from_dict(PatchingElementConfig, {"type": "value_replace"})
    assert element.activation_name is None
    assert element.location == []
    assert element.filter == Filter()
    assert element.group == Group()
    assert element.incremental == Incremental()
    assert element.value == 0.0


def test_from_config_hook_dispatch():
    raw = {"filter": [{"layer": 0}, {"not": {"layer": 1}}], "group": {"layer": [0, 1]}, "incremental": {"layer": [2]}}
    element = from_dict(PatchingElementConfig, {**ELEMENT, **raw})
    assert element.filter == Filter(inclusion_terms=[{"layer": 0}], exclusion_terms=[{"layer": 1}])
    assert element.group == Group(groups={"layer": [0, 1]})
    assert element.incremental == Incremental(values={"layer": [2]})


def test_already_typed_values_pass_through():
    # Code-built objects can be mixed into raw shapes and are taken as-is.
    filter_ = Filter(inclusion_terms=[{"layer": -1}])
    element = from_dict(
        PatchingElementConfig,
        {"activation_name": ModelLayerLocation.RESIDUAL_POST, "filter": filter_, "type": PatchingType.REPLACE},
    )
    assert element.activation_name is ModelLayerLocation.RESIDUAL_POST
    assert element.filter is filter_
    assert element.type is PatchingType.REPLACE

    config = from_dict(PatchingConfig, {"metric": "logit_diff", "patch": [{"source": element, "target": element}]})
    assert config.patch[0].source is element


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown keys for PatchingElementConfig.*typ"):
        from_dict(PatchingElementConfig, {**ELEMENT, "typ": "replace"})


def test_missing_required_key_raises():
    with pytest.raises(ValueError, match="Missing required keys for PatchingElementConfig.*type"):
        from_dict(PatchingElementConfig, {"activation_name": "resid.post"})


def test_non_mapping_for_dataclass_raises():
    with pytest.raises(ValueError, match="Expected a mapping for PatchingElementConfig"):
        from_dict(PatchingElementConfig, "replace")


def test_input_is_not_mutated():
    # YAML anchors alias the same mapping for source and target; deserialization must not touch it.
    raw = {"metric": "logit_diff", "patch": [{"source": ELEMENT, "target": ELEMENT}]}
    snapshot = deepcopy(raw)
    config = from_dict(PatchingConfig, raw)
    assert raw == snapshot
    assert config.patch[0].source == config.patch[0].target


def test_config_factory_stamps_config_id(tmp_path):
    config_yaml = tmp_path / "configs.yaml"
    config_yaml.write_text(
        yaml.safe_dump({"my_patch": {"metric": "logit_diff", "patch": [{"source": ELEMENT, "target": ELEMENT}]}})
    )
    factory = ConfigFactory[PatchingConfig](str(config_yaml))
    config = factory.create("my_patch")
    assert isinstance(config, PatchingConfig)
    assert config.config_id == "my_patch"

    with pytest.raises(KeyError):
        factory.create("missing")
