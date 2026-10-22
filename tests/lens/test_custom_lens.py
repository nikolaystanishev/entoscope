"""A lens a user registers, run end to end the way the built-in ones are."""

import json
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd
import pytest
import yaml

from entoscope.lens.config import LensConfig, LensMethodConfig
from entoscope.lens.input import LensInput
from entoscope.lens.lens import Lens
from entoscope.lens.method.factory import LensMethodFactory
from entoscope.lens.method.logit_lens import LogitLens
from entoscope.lens.method.parameter_lens import ParameterLens, ParameterLensConfig, ProjectionSpace
from entoscope.model.config import Coordinates, Filter, ModelAxis, ModelLayerLocation, ModelLocatorConfig
from entoscope.util.data import from_dict

from ..conftest import TransformerBridge, forbid_model_runs, rendered_figure
from .conftest import HEAD, NEURON, PROMPT, assert_row_reads, mlp_matrix, run_lens, single_row, unembed

TEMPERATURE = 2.0
GRID = {"activation": "resid.post", "location": ["layer", "position"]}


@dataclass(kw_only=True)
class TemperatureLensConfig(LensMethodConfig):
    temperature: float = 1.0


class TemperatureLens(LogitLens[TemperatureLensConfig]):
    """The logit lens with its logits divided by a temperature, which flattens the probabilities."""

    def project(self, activation, index):
        return super().project(activation, index) / self.method_config.temperature


class NegatedParameterLens(ParameterLens):
    """The parameter lens read against the direction of its weight."""

    def project(self, weight, index):
        return super().project(-weight, index)


@dataclass(kw_only=True)
class SourceFileLensConfig(LensMethodConfig):
    path: str


class SourceFileLens(LogitLens[SourceFileLensConfig]):
    """A lens that cannot run without its method config: one of its fields has no default."""


class Reduction(Enum):
    SUM = "sum"
    MEAN = "mean"


@dataclass(kw_only=True)
class ShapedLensConfig(LensMethodConfig):
    """A field of every shape a config holds, ``width`` derived from them rather than given."""

    reduction: Reduction = Reduction.SUM
    window: tuple[int, int] = (0, 1)
    steps: tuple[int, ...] = ()
    labels: set[str] = field(default_factory=set)
    weights: dict[int, float] = field(default_factory=dict)
    checkpoint: Path | None = None
    width: int = field(init=False)

    def __post_init__(self):
        self.width = self.window[1] - self.window[0]


class ShapedLens(LogitLens[ShapedLensConfig]):
    """The logit lens, taking fields of every shape a config holds."""


SHAPED_YAML = {
    "type": "shaped_lens",
    "reduction": "mean",
    "window": [2, 5],
    "steps": [1, 2, 3],
    "labels": ["b", "a"],
    "weights": {0: 0.5},
    "checkpoint": "translators",
}
SHAPED = ShapedLensConfig(
    type="shaped_lens",
    reduction=Reduction.MEAN,
    window=(2, 5),
    steps=(1, 2, 3),
    labels={"a", "b"},
    weights={0: 0.5},
    checkpoint=Path("translators"),
)


@pytest.fixture
def temperature_lens(registries):
    return LensMethodFactory.register("temperature_lens")(TemperatureLens)


@pytest.fixture
def negated_parameter_lens(registries):
    # It extends the parameter lens, so it takes the parameter lens's method config.
    return LensMethodFactory.register("negated_parameter_lens")(NegatedParameterLens)


def _temperature_yaml(**overrides) -> dict:
    return {"method": {"type": "temperature_lens", "temperature": TEMPERATURE}, "source": GRID, **overrides}


def _warm(source: ModelLocatorConfig) -> LensConfig:
    return LensConfig(method=TemperatureLensConfig(type="temperature_lens", temperature=TEMPERATURE), source=source)


def _last_position(activation: str, **coordinate: int) -> ModelLocatorConfig:
    return ModelLocatorConfig(
        coordinates=Coordinates.from_config([{"activation": activation, "layer": -1, "position": -1, **coordinate}])
    )


def _neurons() -> ModelLocatorConfig:
    return ModelLocatorConfig(
        activation=ModelLayerLocation.MLP_POST,
        location=[ModelAxis.LAYER, ModelAxis.NEURON],
        filter=Filter(inclusion_terms=[{ModelAxis.NEURON: NEURON}]),
    )


# --------------------------------------------------------------------------------------------------
# A registered lens is run like a built-in one
# --------------------------------------------------------------------------------------------------
def test_a_registered_lens_reads_every_row_through_its_own_projection(model, activations, temperature_lens):
    _, cache = activations
    grid = ModelLocatorConfig(
        activation=ModelLayerLocation.RESIDUAL_POST, location=[ModelAxis.LAYER, ModelAxis.POSITION]
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


def test_a_registered_lens_leaves_its_method_config_beside_its_results(
    model, config_from_yaml, temperature_lens, tmp_path
):
    config = config_from_yaml(LensConfig, "warm", _temperature_yaml())
    save_dir = tmp_path / "warm"

    run_lens(model, config, save_dir=save_dir)

    saved = json.loads((save_dir / "config.json").read_text())
    assert saved["method"] == {"type": "temperature_lens", "temperature": TEMPERATURE}
    # The saved config reads back as the one the run was made under.
    assert from_dict(LensConfig, saved) == config
    assert len(pd.read_csv(save_dir / "result.csv")) == model.cfg.n_layers * model.to_tokens(PROMPT).shape[1]
    assert rendered_figure(save_dir).title.startswith("Lens - temperature_lens · top-1 (config: warm)")


def test_a_lens_named_by_its_type_alone_takes_its_defaults(model, config_from_yaml, temperature_lens, tmp_path):
    custom = config_from_yaml(LensConfig, "default", {"method": "temperature_lens", "source": GRID})
    built_in = config_from_yaml(LensConfig, "built_in", {"method": "logit_lens", "source": GRID})
    save_dir = tmp_path / "default"

    assert custom.method == TemperatureLensConfig(type="temperature_lens")
    assert run_lens(model, custom, save_dir=save_dir).result.equals(run_lens(model, built_in).result)
    # ...and the run writes those defaults down.
    assert json.loads((save_dir / "config.json").read_text())["method"] == {
        "type": "temperature_lens",
        "temperature": 1.0,
    }


def test_a_plan_names_the_registered_lens(model, monkeypatch, config_from_yaml, temperature_lens):
    forbid_model_runs(monkeypatch, model)
    lens_input = LensInput(config_from_yaml(LensConfig, "warm", _temperature_yaml()), PROMPT)

    plan = Lens(model).plan(lens_input, verbose=False)

    assert plan.describe().splitlines()[:3] == ["Lens plan", "  method: temperature_lens", "  config: warm"]


def test_a_lens_naming_no_method_config_takes_no_fields(model, config_from_yaml, registries):
    LensMethodFactory.register("plain_lens")(LogitLens)

    plain = config_from_yaml(LensConfig, "plain", {"method": "plain_lens", "source": GRID})
    logit = config_from_yaml(LensConfig, "logit", {"method": "logit_lens", "source": GRID})
    warm = {"method": {"type": "plain_lens", "temperature": 2.0}, "source": GRID}

    assert run_lens(model, plain).result.equals(run_lens(model, logit).result)
    with pytest.raises(ValueError, match=r"Unknown keys for LensMethodConfig: \['temperature'\]"):
        config_from_yaml(LensConfig, "warm", warm)


# --------------------------------------------------------------------------------------------------
# A lens that reads only the weights
# --------------------------------------------------------------------------------------------------
def test_a_lens_reading_the_weights_runs_without_a_prompt(model, negated_parameter_lens):
    negated = LensConfig(method=ParameterLensConfig(type="negated_parameter_lens"), source=_neurons())
    flipped = LensConfig(method=ParameterLensConfig(type="parameter_lens", scale=-1.0), source=_neurons())

    assert run_lens(model, negated).result.equals(run_lens(model, flipped).result)
    with pytest.raises(ValueError, match="'negated_parameter_lens' reads the model's weights.* takes no prompt"):
        LensInput(negated, PROMPT)


def test_a_lens_reading_the_weights_has_no_position_axis(config_from_yaml, negated_parameter_lens):
    raw = {"method": "negated_parameter_lens", "source": {"activation": "mlp.post", "location": ["layer", "position"]}}

    with pytest.raises(ValueError, match="'position' is not a location of 'negated_parameter_lens'"):
        config_from_yaml(LensConfig, "positioned", raw)


def test_a_lens_extending_another_takes_its_method_config(model, config_from_yaml, negated_parameter_lens):
    raw = {
        "method": {"type": "negated_parameter_lens", "projection": "embedding", "scale": 2.0},
        "source": {"activation": "mlp.post", "location": ["layer"]},
    }

    lens = LensMethodFactory.create(model, config_from_yaml(LensConfig, "projected", raw))

    assert lens.method_config == ParameterLensConfig(
        type="negated_parameter_lens", projection=ProjectionSpace.EMBEDDING, scale=2.0
    )


# --------------------------------------------------------------------------------------------------
# A lens names its method config once, as its type argument
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# Method configs are data: kept as given, and written down in a form they are read back from
# --------------------------------------------------------------------------------------------------
@pytest.fixture
def shaped_lens(registries):
    return LensMethodFactory.register("shaped_lens")(ShapedLens)


def test_fields_of_every_shape_are_read_from_yaml_and_back_from_the_saved_config(
    model, config_from_yaml, shaped_lens, tmp_path
):
    config = config_from_yaml(LensConfig, "shaped", {"method": SHAPED_YAML, "source": GRID})
    save_dir = tmp_path / "shaped"

    run_lens(model, config, save_dir=save_dir)

    assert config.method == SHAPED
    saved = json.loads((save_dir / "config.json").read_text())
    # JSON has no enum, tuple, set, path or number key, so it holds their raw form - a set in sorted
    # order, so two runs of one config write the same file - and leaves out what is derived...
    assert saved["method"] == {**SHAPED_YAML, "labels": ["a", "b"], "weights": {"0": 0.5}}
    # ...which reads back as the types the method config declares.
    assert from_dict(LensConfig, saved) == config


# --------------------------------------------------------------------------------------------------
# Rejections, as the config is read from a file
# --------------------------------------------------------------------------------------------------
def test_a_lens_nothing_registered_is_rejected_as_its_config_is_read(config_from_yaml):
    registered = r"\['logit_lens', 'parameter_lens', 'tuned_lens'\]"

    with pytest.raises(
        ValueError, match=rf"Unknown lens method: temperature_lens\. Registered lens methods: {registered}"
    ):
        config_from_yaml(LensConfig, "warm", _temperature_yaml())


def test_a_field_the_lens_does_not_have_is_rejected_as_the_config_is_read(config_from_yaml, temperature_lens):
    raw = {"method": {"type": "temperature_lens", "translators": "gpt2"}, "source": GRID}

    with pytest.raises(ValueError, match=r"Unknown keys for TemperatureLensConfig: \['translators'\]"):
        config_from_yaml(LensConfig, "warm", raw)


def test_a_lens_whose_fields_have_no_defaults_needs_them(config_from_yaml, registries):
    LensMethodFactory.register("source_file_lens")(SourceFileLens)

    with pytest.raises(ValueError, match=r"Missing required keys for SourceFileLensConfig: \['path'\]"):
        config_from_yaml(LensConfig, "source_file", {"method": "source_file_lens", "source": GRID})


@pytest.mark.parametrize(
    ("method", "message"),
    [
        ({"temperature": TEMPERATURE}, r"Missing required keys for LensMethodConfig: \['type'\]"),
        (None, "Expected a lens method or a mapping for LensMethodConfig, got None"),
        ([TEMPERATURE], r"Expected a lens method or a mapping for LensMethodConfig, got \[2.0\]"),
    ],
)
def test_a_method_entry_naming_no_type_is_rejected_as_the_config_is_read(config_from_yaml, method, message):
    with pytest.raises(ValueError, match=message):
        config_from_yaml(LensConfig, "untyped", {"method": method, "source": GRID})


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"window": [2]}, r"Expected 2 values for tuple\[int, int\], got \[2\]"),
        ({"window": 2}, r"Expected a list for tuple\[int, int\], got 2"),
        ({"labels": "a"}, r"Expected a list for set\[str\], got 'a'"),
    ],
)
def test_a_field_of_the_wrong_shape_is_rejected_as_the_config_is_read(config_from_yaml, shaped_lens, fields, message):
    with pytest.raises(ValueError, match=message):
        config_from_yaml(LensConfig, "shaped", {"method": {"type": "shaped_lens", **fields}, "source": GRID})


def test_a_lens_is_named_by_a_non_empty_string(config_from_yaml, registries):
    with pytest.raises(ValueError, match="A lens method is named by a non-empty string, got ''"):
        config_from_yaml(LensConfig, "unnamed", {"method": "", "source": GRID})
    with pytest.raises(ValueError, match="A lens method is named by a non-empty string"):
        LensMethodFactory.register("")(TemperatureLens)


def test_only_a_lens_can_be_registered_as_one(registries):
    # Checked as it is registered, rather than at the first run that names it.
    with pytest.raises(TypeError, match="A lens method must subclass"):
        LensMethodFactory.register("not_a_lens")(TemperatureLensConfig)

    assert "not_a_lens" not in LensMethodFactory.registered()


# --------------------------------------------------------------------------------------------------
# Driven from an experiment script that defines the lens itself - the way a user runs one
# --------------------------------------------------------------------------------------------------
CUSTOM_LENS_SCRIPT = """\
from dataclasses import dataclass

import torch
from transformer_lens.model_bridge import TransformerBridge
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from entoscope.lens.config import LensConfig, LensMethodConfig
from entoscope.lens.input import LensInput
from entoscope.lens.lens import Lens
from entoscope.lens.method.factory import LensMethodFactory
from entoscope.lens.method.logit_lens import LogitLens
from entoscope.operation.config import ConfigFactory
from entoscope.script import experiment


@dataclass(kw_only=True)
class TemperatureLensConfig(LensMethodConfig):
    temperature: float = 1.0


@LensMethodFactory.register("temperature_lens")
class TemperatureLens(LogitLens[TemperatureLensConfig]):
    def project(self, activation, index):
        return super().project(activation, index) / self.method_config.temperature


@dataclass
class Config:
    config_path: str
    config_id: str
    save_dir: str


@experiment
def main(config: Config):
    torch.manual_seed(0)
    hf_model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=8, n_head=4, n_inner=6, n_positions=16)).eval()
    model = TransformerBridge.boot_transformers(
        "tiny", hf_model=hf_model, tokenizer=AutoTokenizer.from_pretrained("gpt2"), device="cpu"
    )
    lens_config = ConfigFactory[LensConfig](config.config_path).create(config.config_id)
    Lens(model).apply(LensInput(lens_config, "Paris is located in"), save_dir=config.save_dir, verbose=False)
    print(lens_config.method)
"""


@pytest.mark.skipif(TransformerBridge is None, reason="the script loads its model as a TransformerBridge")
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
    assert (
        result.stdout.strip().splitlines()[-1]
        == f"TemperatureLensConfig(type='temperature_lens', temperature={TEMPERATURE})"
    )
    saved = json.loads((save_dir / "config.json").read_text())
    assert (saved["method"], saved["config_id"]) == ({"type": "temperature_lens", "temperature": TEMPERATURE}, "warm")
    assert (save_dir / "result.csv").exists()
