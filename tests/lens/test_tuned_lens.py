"""The tuned lens, run end to end against real translators.

Two sources of translators are used: the pretrained GPT-2 ones from the tuned-lens hub space (the
path a run takes in practice), and translators saved into a local directory with weights chosen so
the translated residual stays hand-checkable — the documented ``translators: <directory>`` form.
"""

import pytest
import torch
from tuned_lens.nn.lenses import TunedLens as PretrainedTunedLens
from tuned_lens.nn.lenses import TunedLensConfig

from intex.lens.config import LensConfig, LensType
from intex.lens.input import LensInput
from intex.lens.lens import Lens
from intex.model.config import Filter, ModelLayerLocation, ModelLocation, ModelLocatorConfig
from intex.model.unembed import build_unembed

from ..conftest import build_tiny_model
from .conftest import (
    HEAD,
    NEURON,
    PROMPT,
    assert_row_reads,
    coordinates_config,
    declarative_config,
    lens_rows,
    single_row,
    unembed,
)

SCALE = 0.1  # translator i scales the residual by i * SCALE


def _write_translators(model, directory, *, scale: float = 0.0, bias: float = 0.0, **config_overrides) -> str:
    settings = {
        "base_model_name_or_path": "gpt2",
        "d_model": model.cfg.d_model,
        "num_hidden_layers": model.cfg.n_layers,
        **config_overrides,
    }
    lens = PretrainedTunedLens(build_unembed(model), TunedLensConfig(**settings))
    with torch.no_grad():
        for i, translator in enumerate(lens.layer_translators):
            translator.weight.copy_(i * scale * torch.eye(settings["d_model"]))
            translator.bias.copy_(i * bias * torch.ones(settings["d_model"]))
    lens.save(str(directory))
    return str(directory)


@pytest.fixture(scope="module")
def translators(model, tmp_path_factory):
    """Local translator directories, keyed by how their weights were chosen."""
    root = tmp_path_factory.mktemp("translators")
    return {
        "zero": _write_translators(model, root / "zero"),
        "scaled": _write_translators(model, root / "scaled", scale=SCALE),
        "scaled_with_bias": _write_translators(model, root / "scaled_with_bias", scale=SCALE, bias=1.0),
        "wider_model": _write_translators(model, root / "wider_model", d_model=model.cfg.d_model + 1),
        "deeper_model": _write_translators(model, root / "deeper_model", num_hidden_layers=model.cfg.n_layers + 1),
    }


def tuned_config(*coordinates: dict, resource: str | None = None, top_k: int = 1) -> LensConfig:
    config = coordinates_config(*coordinates, lens_type=LensType.TUNED_LENS, top_k=top_k)
    config.translators = resource
    return config


# --------------------------------------------------------------------------------------------------
# Translators that do nothing — the tuned lens falls back to the logit lens
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "coordinate",
    [
        {"activation": "embed", "position": -1},
        {"activation": "resid.post", "layer": 3, "position": -1},
        {"activation": "attn.out", "layer": 3, "position": -1},
        {"activation": "mlp.post", "layer": 3, "position": -1, "neuron": NEURON},
        {"activation": "attn.z", "layer": 3, "position": -1, "head": HEAD},
    ],
)
def test_zero_translators_read_exactly_like_the_logit_lens(model, translators, coordinate):
    tuned = single_row(model, tuned_config(coordinate, resource=translators["zero"]))
    logit = single_row(model, coordinates_config(coordinate))

    assert tuned["token_0"] == logit["token_0"]
    assert tuned["probability_0"] == pytest.approx(logit["probability_0"], rel=1e-5)


# --------------------------------------------------------------------------------------------------
# Translators are indexed by residual state: state l is resid.pre of layer l, and everything layer l
# writes lands in state l + 1
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("coordinate", "hook", "translator_index"),
    [
        ({"activation": "resid.pre", "layer": 3, "position": -1}, "blocks.3.hook_resid_pre", 3),
        ({"activation": "resid.post", "layer": 3, "position": -1}, "blocks.3.hook_resid_post", 4),
        ({"activation": "resid.mid", "layer": 3, "position": -1}, "blocks.3.hook_resid_mid", 4),
    ],
)
def test_a_residual_state_is_translated_by_the_translator_of_its_state(
    model, activations, translators, coordinate, hook, translator_index
):
    _, cache = activations
    row = single_row(model, tuned_config(coordinate, resource=translators["scaled"]))

    residual = cache[hook][:, -1]
    assert_row_reads(model, row, unembed(model, (1 + translator_index * SCALE) * residual))


def test_a_residual_state_is_translated_with_the_bias(model, activations, translators):
    _, cache = activations
    row = single_row(
        model,
        tuned_config({"activation": "resid.pre", "layer": 2, "position": -1}, resource=translators["scaled_with_bias"]),
    )

    residual = cache["blocks.2.hook_resid_pre"][:, -1]
    assert_row_reads(model, row, unembed(model, (1 + 2 * SCALE) * residual + 2 * torch.ones(model.cfg.d_model)))


@pytest.mark.parametrize(
    ("coordinate", "hook"),
    [
        ({"activation": "attn.out", "layer": 1, "position": -1}, "blocks.1.hook_attn_out"),
        ({"activation": "mlp.out", "layer": 1, "position": -1}, "blocks.1.hook_mlp_out"),
    ],
)
def test_a_component_write_is_translated_without_the_bias(model, activations, translators, coordinate, hook):
    # The bias belongs to the whole state, not to the share of it one component wrote.
    _, cache = activations
    row = single_row(model, tuned_config(coordinate, resource=translators["scaled_with_bias"]))

    assert_row_reads(model, row, unembed(model, (1 + 2 * SCALE) * cache[hook][:, -1]))


def test_the_embedding_is_translated_by_the_first_translator(model, activations, translators):
    _, cache = activations
    row = single_row(
        model, tuned_config({"activation": "embed", "position": -1}, resource=translators["scaled_with_bias"])
    )

    # Translator 0 scales by 0 and shifts by 0, so the embedding comes through untouched.
    assert_row_reads(model, row, unembed(model, cache["hook_embed"][:, -1]))


def test_the_last_residual_state_is_left_untranslated(model, activations, translators):
    logits, _ = activations
    row = single_row(
        model, tuned_config({"activation": "resid.post", "layer": -1, "position": -1}, resource=translators["scaled"])
    )

    # The model unembeds that state itself, so the tuned lens has nothing left to map.
    assert_row_reads(model, row, logits[0, -1])


def test_a_head_output_is_lifted_before_it_is_translated(model, activations, translators):
    _, cache = activations
    layer = 3
    row = single_row(
        model,
        tuned_config(
            {"activation": "attn.z", "layer": layer, "position": -1, "head": HEAD}, resource=translators["scaled"]
        ),
    )

    written = cache[f"blocks.{layer}.attn.hook_z"][:, -1, HEAD] @ model.blocks[layer].attn.W_O[HEAD]
    assert_row_reads(model, row, unembed(model, (1 + (layer + 1) * SCALE) * written))


# --------------------------------------------------------------------------------------------------
# The pretrained translators of the hub space
# --------------------------------------------------------------------------------------------------
def test_a_declarative_run_reads_every_layer_through_its_own_translator(model, activations):
    logits, _ = activations
    config = declarative_config(
        ModelLayerLocation.RESIDUAL_POST,
        [ModelLocation.LAYER, ModelLocation.POSITION],
        lens_type=LensType.TUNED_LENS,
        filter=Filter.from_config([{"position": -1}]),
    )
    config.translators = "gpt2"

    result = lens_rows(model, config)

    assert len(result) == model.cfg.n_layers
    # The last state is the one the model reads out itself...
    assert_row_reads(model, result.iloc[-1], logits[0, -1])
    # ...while the earlier ones are mapped onto it, which the plain logit lens does not do.
    plain = lens_rows(model, coordinates_config({"activation": "resid.post", "layer": 0, "position": -1}))
    assert result.iloc[0]["probability_0"] != pytest.approx(plain.iloc[0]["probability_0"], rel=1e-3)


def test_the_translators_default_to_the_models_own_name(model):
    coordinate = {"activation": "resid.post", "layer": 5, "position": -1}

    named = single_row(model, tuned_config(coordinate, resource="gpt2"))
    default = single_row(model, tuned_config(coordinate))

    assert named["token_0"] == default["token_0"]
    assert named["probability_0"] == pytest.approx(default["probability_0"])


def test_a_model_without_a_name_has_to_name_its_translators(model):
    nameless = build_tiny_model(tokenizer_name=None, model_name=None)
    nameless.set_tokenizer(model.tokenizer)
    config = tuned_config({"activation": "resid.post", "layer": 0, "position": -1})

    with pytest.raises(ValueError, match="Name the translators explicitly"):
        Lens(nameless).apply(LensInput(config, PROMPT), verbose=False)


@pytest.mark.parametrize("directory", ["wider_model", "deeper_model"])
def test_translators_trained_for_another_model_are_rejected(model, translators, directory):
    config = tuned_config({"activation": "resid.post", "layer": 0, "position": -1}, resource=translators[directory])

    with pytest.raises(ValueError, match="was trained for a model with"):
        Lens(model).apply(LensInput(config, PROMPT), verbose=False)


# --------------------------------------------------------------------------------------------------
# Rejections
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("activation_name", ["attn.q", "attn.k"])
def test_queries_and_keys_have_no_tuned_projection(model, translators, activation_name):
    config = tuned_config(
        {"activation": activation_name, "layer": 0, "head": 0, "position": -1}, resource=translators["zero"]
    )

    with pytest.raises(ValueError, match="Tuned lens cannot project.*use the logit lens"):
        Lens(model).apply(LensInput(config, PROMPT), verbose=False)


def test_an_activation_without_a_vocabulary_projection_raises(model, translators):
    config = tuned_config(
        {"activation": "attn.pattern", "layer": 0, "head": 0, "position": -1}, resource=translators["zero"]
    )

    with pytest.raises(ValueError, match="Tuned lens cannot project.*map onto the vocabulary"):
        Lens(model).apply(LensInput(config, PROMPT), verbose=False)


def test_translators_on_another_lens_type_raise(config_from_yaml):
    raw = {"type": "logit_lens", "translators": "gpt2", "source": {"activation": "resid.post", "location": ["layer"]}}

    with pytest.raises(ValueError, match="'translators' is a tuned lens setting"):
        config_from_yaml(LensConfig, "tuned", raw)


def test_a_yaml_tuned_lens_config_runs(model, config_from_yaml):
    raw = {
        "type": "tuned_lens",
        "translators": "gpt2",
        "top_k": 2,
        "source": {"coordinates": [{"activation": "resid.post", "layer": -1, "position": -1}]},
    }
    config = config_from_yaml(LensConfig, "my_tuned_lens", raw)

    assert config == LensConfig(
        type=LensType.TUNED_LENS,
        translators="gpt2",
        top_k=2,
        source=ModelLocatorConfig(
            coordinates=[
                {
                    ModelLocation.ACTIVATION: ModelLayerLocation.RESIDUAL_POST,
                    ModelLocation.LAYER: -1,
                    ModelLocation.POSITION: -1,
                }
            ]
        ),
        config_id="my_tuned_lens",
    )
    assert len(lens_rows(model, config)) == 1


def test_translators_default_to_none(config_from_yaml):
    raw = {"type": "tuned_lens", "source": {"activation": "resid.post", "location": ["layer"]}}

    assert config_from_yaml(LensConfig, "my_tuned_lens", raw).translators is None
