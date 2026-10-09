"""A lens a user registers, run end to end the way the built-in ones are."""

import json
import subprocess
import sys
from dataclasses import dataclass

import pandas as pd
import pytest
import yaml

from intex.lens.config import LensConfig, LensSettings
from intex.lens.input import LensInput
from intex.lens.lens import Lens
from intex.lens.method.factory import LensMethodFactory
from intex.lens.method.logit_lens import LogitLens
from intex.lens.method.parameter_lens import ParameterLens, ParameterLensSettings, ProjectionSpace
from intex.model.config import Filter, ModelLayerLocation, ModelLocation, ModelLocatorConfig
from intex.util.data import from_dict

from ..conftest import forbid_model_runs, rendered_figure
from .conftest import HEAD, NEURON, PROMPT, assert_row_reads, mlp_matrix, run_lens, single_row, unembed

TEMPERATURE = 2.0
GRID = {"activation": "resid.post", "location": ["layer", "position"]}


@dataclass(kw_only=True)
class TemperatureLensSettings(LensSettings):
    temperature: float = 1.0


class TemperatureLens(LogitLens):
    """The logit lens with its logits divided by a temperature, which flattens the probabilities."""

    settings_class = TemperatureLensSettings

    def project(self, activation, index):
        return super().project(activation, index) / self.settings.temperature


class NegatedParameterLens(ParameterLens):
    """The parameter lens read against the direction of its weight."""

    def project(self, weight, index):
        return super().project(-weight, index)


@dataclass(kw_only=True)
class SourceFileSettings(LensSettings):
    path: str


class SourceFileLens(LogitLens):
    """A lens that cannot run without its settings: one of them has no default."""

    settings_class = SourceFileSettings


@pytest.fixture
def temperature_lens(registries):
    return LensMethodFactory.register("temperature_lens")(TemperatureLens)


@pytest.fixture
def negated_parameter_lens(registries):
    # It extends the parameter lens, so it takes the parameter lens's settings.
    return LensMethodFactory.register("negated_parameter_lens")(NegatedParameterLens)


def _temperature_yaml(**overrides) -> dict:
    return {"method": "temperature_lens", "settings": {"temperature": TEMPERATURE}, "source": GRID, **overrides}


def _warm(source: ModelLocatorConfig) -> LensConfig:
    return LensConfig(
        method="temperature_lens", settings=TemperatureLensSettings(temperature=TEMPERATURE), source=source
    )


def _last_position(activation: str, **coordinate: int) -> ModelLocatorConfig:
    return ModelLocatorConfig(coordinates=[{"activation": activation, "layer": -1, "position": -1, **coordinate}])


def _neurons() -> ModelLocatorConfig:
    return ModelLocatorConfig(
        activation=ModelLayerLocation.MLP_POST,
        location=[ModelLocation.LAYER, ModelLocation.NEURON],
        filter=Filter(inclusion_terms=[{ModelLocation.NEURON: NEURON}]),
    )


# --------------------------------------------------------------------------------------------------
# A registered lens is run like a built-in one
# --------------------------------------------------------------------------------------------------
def test_a_registered_lens_reads_every_row_through_its_own_projection(model, activations, temperature_lens):
    _, cache = activations
    grid = ModelLocatorConfig(
        activation=ModelLayerLocation.RESIDUAL_POST, location=[ModelLocation.LAYER, ModelLocation.POSITION]
    )

    result = run_lens(model, _warm(grid)).result

    assert len(result) == model.cfg.n_layers * model.to_tokens(PROMPT).shape[1]
    for _, row in result.iterrows():
        residual = cache[f"blocks.{row['source_layer']}.hook_resid_post"][:, row["source_position"]]
        assert_row_reads(model, row, unembed(model, residual) / TEMPERATURE)


def test_the_lens_it_extends_hands_it_every_kind_of_activation(model, activations, temperature_lens):
    # One class per lens: the override applies whether the element is read from the residual stream,
    # lifted out of an MLP neuron, or lifted out of an attention head.
    _, cache = activations
    layer = model.cfg.n_layers - 1

    residual = single_row(model, _warm(_last_position("resid.post")))
    assert_row_reads(model, residual, unembed(model, cache[f"blocks.{layer}.hook_resid_post"][:, -1]) / TEMPERATURE)

    neuron = single_row(model, _warm(_last_position("mlp.post", neuron=NEURON)))
    activation = cache[f"blocks.{layer}.mlp.hook_post"][:, -1, NEURON]
    written = activation.unsqueeze(-1) * mlp_matrix(model, layer, "W_out")[NEURON]
    assert_row_reads(model, neuron, unembed(model, written) / TEMPERATURE)

    head = single_row(model, _warm(_last_position("attn.z", head=HEAD)))
    activation = cache[f"blocks.{layer}.attn.hook_z"][:, -1, HEAD]
    assert_row_reads(model, head, unembed(model, activation @ model.blocks[layer].attn.W_O[HEAD]) / TEMPERATURE)


def test_a_yaml_config_is_built_as_the_code_built_one(config_from_yaml, temperature_lens):
    config = config_from_yaml(LensConfig, "warm", _temperature_yaml())

    in_code = _warm(from_dict(ModelLocatorConfig, GRID))
    in_code.config_id = "warm"
    assert config == in_code


def test_a_registered_lens_leaves_its_name_and_settings_beside_its_results(
    model, config_from_yaml, temperature_lens, tmp_path
):
    config = config_from_yaml(LensConfig, "warm", _temperature_yaml())
    save_dir = tmp_path / "warm"

    run_lens(model, config, save_dir=save_dir)

    saved = json.loads((save_dir / "config.json").read_text())
    assert (saved["method"], saved["settings"]) == ("temperature_lens", {"temperature": TEMPERATURE})
    # The saved config reads back as the one the run was made under.
    assert from_dict(LensConfig, saved) == config
    assert len(pd.read_csv(save_dir / "result.csv")) == model.cfg.n_layers * model.to_tokens(PROMPT).shape[1]
    assert rendered_figure(save_dir).title.startswith("Lens — temperature_lens · top-1 (config: warm)")


def test_settings_left_out_are_the_lens_defaults(model, config_from_yaml, temperature_lens, tmp_path):
    custom = config_from_yaml(LensConfig, "default", {"method": "temperature_lens", "source": GRID})
    built_in = config_from_yaml(LensConfig, "built_in", {"method": "logit_lens", "source": GRID})
    save_dir = tmp_path / "default"

    assert custom.settings == TemperatureLensSettings()
    assert run_lens(model, custom, save_dir=save_dir).result.equals(run_lens(model, built_in).result)
    # ...and the run writes those defaults down.
    assert json.loads((save_dir / "config.json").read_text())["settings"] == {"temperature": 1.0}


def test_a_plan_names_the_registered_lens(model, monkeypatch, config_from_yaml, temperature_lens):
    forbid_model_runs(monkeypatch, model)
    lens_input = LensInput(config_from_yaml(LensConfig, "warm", _temperature_yaml()), PROMPT)

    plan = Lens(model).plan(lens_input, verbose=False)

    assert plan.describe().splitlines()[0] == "Lens plan — temperature_lens (config: warm)"


def test_a_lens_naming_no_settings_takes_none(model, config_from_yaml, registries):
    LensMethodFactory.register("plain_lens")(LogitLens)

    plain = config_from_yaml(LensConfig, "plain", {"method": "plain_lens", "source": GRID})
    logit = config_from_yaml(LensConfig, "logit", {"method": "logit_lens", "source": GRID})
    warm = {"method": "plain_lens", "settings": {"temperature": 2.0}, "source": GRID}

    assert run_lens(model, plain).result.equals(run_lens(model, logit).result)
    with pytest.raises(ValueError, match=r"Unknown keys for LensSettings: \['temperature'\]"):
        config_from_yaml(LensConfig, "warm", warm)


# --------------------------------------------------------------------------------------------------
# A lens that reads only the weights
# --------------------------------------------------------------------------------------------------
def test_a_lens_reading_the_weights_runs_without_a_prompt(model, negated_parameter_lens):
    negated = LensConfig(method="negated_parameter_lens", source=_neurons())
    flipped = LensConfig(method="parameter_lens", settings=ParameterLensSettings(scale=-1.0), source=_neurons())

    assert run_lens(model, negated).result.equals(run_lens(model, flipped).result)
    with pytest.raises(ValueError, match="'negated_parameter_lens' reads the model's weights.* takes no prompt"):
        LensInput(negated, PROMPT)


def test_a_lens_reading_the_weights_has_no_position_axis(config_from_yaml, negated_parameter_lens):
    raw = {"method": "negated_parameter_lens", "source": {"activation": "mlp.post", "location": ["layer", "position"]}}

    with pytest.raises(ValueError, match="'position' is not a location of 'negated_parameter_lens'"):
        LensInput(config_from_yaml(LensConfig, "positioned", raw))


def test_a_lens_extending_another_takes_its_settings(model, config_from_yaml, negated_parameter_lens):
    raw = {"method": "negated_parameter_lens", "settings": {"projection": "embedding", "scale": 2.0}, "source": GRID}

    lens = LensMethodFactory.create(config_from_yaml(LensConfig, "projected", raw), model)

    assert lens.settings == ParameterLensSettings(projection=ProjectionSpace.EMBEDDING, scale=2.0)


# --------------------------------------------------------------------------------------------------
# Rejections, as the config is built
# --------------------------------------------------------------------------------------------------
def test_a_lens_nothing_registered_is_rejected_as_its_config_is_built(config_from_yaml):
    registered = r"\['logit_lens', 'parameter_lens', 'tuned_lens'\]"

    with pytest.raises(
        ValueError, match=rf"Unknown lens method: temperature_lens\. Registered lens methods: {registered}"
    ):
        config_from_yaml(LensConfig, "warm", _temperature_yaml())


def test_a_setting_the_lens_does_not_have_is_rejected_as_the_config_is_built(temperature_lens):
    with pytest.raises(ValueError, match=r"Unknown keys for TemperatureLensSettings: \['translators'\]"):
        LensConfig(method="temperature_lens", settings={"translators": "gpt2"}, source=_last_position("resid.post"))


def test_a_lens_whose_settings_have_no_defaults_needs_them(registries):
    LensMethodFactory.register("source_file_lens")(SourceFileLens)

    with pytest.raises(ValueError, match=r"Missing required keys for SourceFileSettings: \['path'\]"):
        LensConfig(method="source_file_lens", source=_last_position("resid.post"))


# --------------------------------------------------------------------------------------------------
# Driven from an experiment script that defines the lens itself - the way a user runs one
# --------------------------------------------------------------------------------------------------
CUSTOM_LENS_SCRIPT = """\
from dataclasses import dataclass

from transformer_lens import HookedTransformer, HookedTransformerConfig

from intex.lens.config import LensConfig, LensSettings
from intex.lens.input import LensInput
from intex.lens.lens import Lens
from intex.lens.method.factory import LensMethodFactory
from intex.lens.method.logit_lens import LogitLens
from intex.operation.config import ConfigFactory
from intex.script import experiment


@dataclass(kw_only=True)
class TemperatureLensSettings(LensSettings):
    temperature: float = 1.0


@LensMethodFactory.register("temperature_lens")
class TemperatureLens(LogitLens):
    settings_class = TemperatureLensSettings

    def project(self, activation, index):
        return super().project(activation, index) / self.settings.temperature


@dataclass
class Config:
    config_path: str
    config_id: str
    save_dir: str


@experiment
def main(config: Config):
    model = HookedTransformer(
        HookedTransformerConfig(
            n_layers=2, d_model=8, n_ctx=16, d_head=4, n_heads=4, d_mlp=6, d_vocab=50257,
            act_fn="gelu", tokenizer_name="gpt2", seed=0,
        )
    )
    lens_config = ConfigFactory[LensConfig](config.config_path).create(config.config_id)
    Lens(model).apply(LensInput(lens_config, "Paris is located in"), save_dir=config.save_dir, verbose=False)
    print(lens_config.settings)
"""


def test_a_lens_registered_by_an_experiment_script_runs_from_its_yaml_config(tmp_path):
    script = tmp_path / "custom_lens_script.py"
    script.write_text(CUSTOM_LENS_SCRIPT)
    config_path = tmp_path / "configs.yaml"
    config_path.write_text(yaml.safe_dump({"warm": _temperature_yaml()}))
    save_dir = tmp_path / "warm"

    result = subprocess.run(
        [sys.executable, str(script), "--config-path", str(config_path)]
        + ["--config-id", "warm", "--save-dir", str(save_dir)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == f"TemperatureLensSettings(temperature={TEMPERATURE})"
    saved = json.loads((save_dir / "config.json").read_text())
    assert (saved["method"], saved["settings"], saved["config_id"]) == (
        "temperature_lens",
        {"temperature": TEMPERATURE},
        "warm",
    )
    assert (save_dir / "result.csv").exists()
