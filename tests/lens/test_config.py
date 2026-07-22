import pytest
import yaml

from intex.lens.config import LensConfig, LensType
from intex.model.config import Filter, ModelLayerLocation, ModelLocation, ModelLocatorConfig
from intex.operation.config import ConfigFactory, from_dict

SOURCE = {
    "activation": "resid.post",
    "location": ["layer", "position"],
    "filter": [{"layer": -1}],
}

LENS = {"type": "logit_lens", "source": SOURCE}


def source_config() -> ModelLocatorConfig:
    return ModelLocatorConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelLocation.LAYER, ModelLocation.POSITION],
        filter=Filter(inclusion_terms=[{ModelLocation.LAYER: -1}]),
    )


def test_yaml_shape_equals_code_built_config():
    config = from_dict(LensConfig, LENS)
    assert config == LensConfig(type=LensType.LOGIT_LENS, source=source_config())


def test_enum_coercion_by_value():
    config = from_dict(LensConfig, LENS)
    assert config.type is LensType.LOGIT_LENS
    assert config.source.activation is ModelLayerLocation.RESIDUAL_POST
    assert config.source.location == [ModelLocation.LAYER, ModelLocation.POSITION]


def test_top_k_defaults_to_one():
    assert from_dict(LensConfig, LENS).top_k == 1


def test_top_k_explicit():
    assert from_dict(LensConfig, {**LENS, "top_k": 5}).top_k == 5


def test_invalid_type_raises():
    with pytest.raises(ValueError):
        from_dict(LensConfig, {**LENS, "type": "attention_lens"})


def test_missing_type_raises():
    with pytest.raises(ValueError, match="Missing required keys for LensConfig.*type"):
        from_dict(LensConfig, {"source": SOURCE})


def test_missing_source_raises():
    with pytest.raises(ValueError, match="Missing required keys for LensConfig.*source"):
        from_dict(LensConfig, {"type": "logit_lens"})


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown keys for LensConfig.*target"):
        from_dict(LensConfig, {**LENS, "target": SOURCE})


def test_tuned_lens_yaml_shape():
    config = from_dict(LensConfig, {"type": "tuned_lens", "translators": "gpt2", "source": SOURCE})
    assert config == LensConfig(type=LensType.TUNED_LENS, translators="gpt2", source=source_config())


def test_translators_defaults_to_none():
    assert from_dict(LensConfig, {**LENS, "type": "tuned_lens"}).translators is None


def test_translators_on_a_logit_lens_raises():
    with pytest.raises(ValueError, match="'translators' is a tuned lens setting"):
        from_dict(LensConfig, {**LENS, "translators": "gpt2"})


def test_config_factory_stamps_config_id(tmp_path):
    config_yaml = tmp_path / "configs.yaml"
    config_yaml.write_text(yaml.safe_dump({"my_lens": LENS}))
    factory = ConfigFactory[LensConfig](str(config_yaml))
    config = factory.create("my_lens")
    assert isinstance(config, LensConfig)
    assert config.config_id == "my_lens"

    with pytest.raises(KeyError):
        factory.create("missing")
