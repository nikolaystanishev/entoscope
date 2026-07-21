import pytest
import yaml

from intex.circuit.config import CircuitConfig, CircuitMethodType
from intex.operation.config import ConfigFactory, from_dict

IFR = {"type": "information_flow_routes"}


def test_yaml_shape_equals_code_built_config():
    config = from_dict(CircuitConfig, IFR)
    assert config == CircuitConfig(type=CircuitMethodType.INFORMATION_FLOW_ROUTES)


def test_enum_coercion_by_value():
    assert from_dict(CircuitConfig, IFR).type is CircuitMethodType.INFORMATION_FLOW_ROUTES


def test_defaults():
    config = from_dict(CircuitConfig, IFR)
    assert config.threshold == 0.04
    assert config.path_threshold is None
    assert config.prediction_positions == [-1]


def test_explicit_values():
    config = from_dict(
        CircuitConfig,
        {**IFR, "threshold": 0.1, "path_threshold": 0.2, "prediction_positions": [-1, 0]},
    )
    assert config.threshold == 0.1
    assert config.path_threshold == 0.2
    assert config.prediction_positions == [-1, 0]


def test_resolved_path_threshold_defaults_to_threshold():
    assert from_dict(CircuitConfig, {**IFR, "threshold": 0.07}).resolved_path_threshold() == 0.07
    explicit = from_dict(CircuitConfig, {**IFR, "threshold": 0.07, "path_threshold": 0.01})
    assert explicit.resolved_path_threshold() == 0.01


def test_invalid_type_raises():
    with pytest.raises(ValueError):
        from_dict(CircuitConfig, {"type": "acdc"})


def test_missing_type_raises():
    with pytest.raises(ValueError, match="Missing required keys for CircuitConfig.*type"):
        from_dict(CircuitConfig, {"threshold": 0.04})


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown keys for CircuitConfig.*bogus"):
        from_dict(CircuitConfig, {**IFR, "bogus": 1})


def test_config_factory_stamps_config_id(tmp_path):
    config_yaml = tmp_path / "configs.yaml"
    config_yaml.write_text(yaml.safe_dump({"my_circuit": IFR}))
    factory = ConfigFactory[CircuitConfig](str(config_yaml))
    config = factory.create("my_circuit")
    assert isinstance(config, CircuitConfig)
    assert config.config_id == "my_circuit"

    with pytest.raises(KeyError):
        factory.create("missing")
