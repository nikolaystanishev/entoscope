import pytest
import yaml

from intex.model.config import Filter, ModelLayerLocation, ModelLocation, ModelLocatorConfig
from intex.operation.config import ConfigFactory, from_dict
from intex.probe.config import PoolingType, ProbeConfig, ProbeMetric, ProbeType

SOURCE = {
    "activation": "resid.post",
    "location": ["layer"],
    "filter": [{"layer": -1}],
}

PROBE = {"type": "linear", "pool": {"position": "mean"}, "metric": ["accuracy", "auroc"], "source": SOURCE}


def source_config() -> ModelLocatorConfig:
    return ModelLocatorConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelLocation.LAYER],
        filter=Filter(inclusion_terms=[{ModelLocation.LAYER: -1}]),
    )


def test_yaml_shape_equals_code_built_config():
    config = from_dict(ProbeConfig, PROBE)
    assert config == ProbeConfig(
        type=ProbeType.LINEAR,
        pool={ModelLocation.POSITION: PoolingType.MEAN},
        metric=[ProbeMetric.ACCURACY, ProbeMetric.AUROC],
        source=source_config(),
    )


def test_enum_coercion_by_value():
    config = from_dict(ProbeConfig, PROBE)
    assert config.type is ProbeType.LINEAR
    assert config.pool == {ModelLocation.POSITION: PoolingType.MEAN}
    assert all(isinstance(key, ModelLocation) and isinstance(value, PoolingType) for key, value in config.pool.items())
    assert config.metric == [ProbeMetric.ACCURACY, ProbeMetric.AUROC]


def test_pool_defaults_to_empty():
    assert from_dict(ProbeConfig, {key: value for key, value in PROBE.items() if key != "pool"}).pool == {}


def test_scalar_pool_raises():
    with pytest.raises(ValueError, match="Expected a mapping"):
        from_dict(ProbeConfig, {**PROBE, "pool": "mean"})


def test_null_pool_raises():
    with pytest.raises(ValueError, match="Expected a mapping"):
        from_dict(ProbeConfig, {**PROBE, "pool": None})


@pytest.mark.parametrize("location", ["layer", "neuron"])
def test_unpoolable_location_raises(location):
    with pytest.raises(ValueError, match="cannot be pooled"):
        from_dict(ProbeConfig, {**PROBE, "pool": {location: "mean"}})


def test_training_defaults():
    config = from_dict(ProbeConfig, PROBE)
    assert config.max_iter == 1000
    assert config.seed == 42


def test_training_explicit():
    config = from_dict(ProbeConfig, {**PROBE, "max_iter": 50, "seed": 7})
    assert config.max_iter == 50
    assert config.seed == 7


def test_invalid_type_raises():
    with pytest.raises(ValueError):
        from_dict(ProbeConfig, {**PROBE, "type": "mlp"})


def test_invalid_pool_raises():
    with pytest.raises(ValueError):
        from_dict(ProbeConfig, {**PROBE, "pool": {"position": "median"}})
    with pytest.raises(ValueError):
        from_dict(ProbeConfig, {**PROBE, "pool": {"attention": "mean"}})


def test_invalid_metric_raises():
    with pytest.raises(ValueError):
        from_dict(ProbeConfig, {**PROBE, "metric": ["accuracy", "mse"]})


def test_non_list_metric_raises():
    with pytest.raises(ValueError, match="Expected a list"):
        from_dict(ProbeConfig, {**PROBE, "metric": "accuracy"})


def test_missing_keys_raise():
    with pytest.raises(ValueError, match="Missing required keys for ProbeConfig.*metric.*source.*type"):
        from_dict(ProbeConfig, {"pool": "mean"})


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown keys for ProbeConfig.*target"):
        from_dict(ProbeConfig, {**PROBE, "target": SOURCE})


def test_config_factory_stamps_config_id(tmp_path):
    config_yaml = tmp_path / "configs.yaml"
    config_yaml.write_text(yaml.safe_dump({"my_probe": PROBE}))
    factory = ConfigFactory[ProbeConfig](str(config_yaml))
    config = factory.create("my_probe")
    assert isinstance(config, ProbeConfig)
    assert config.config_id == "my_probe"

    with pytest.raises(KeyError):
        factory.create("missing")
