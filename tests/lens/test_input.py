import pandas as pd
import pytest
import torch

from intex.lens.config import LensConfig, LensType
from intex.lens.input import LensInput
from intex.model.config import ModelLayerLocation, ModelLocation, ModelLocatorConfig

PROMPT = "The capital of France is"


def _declarative_config() -> LensConfig:
    return LensConfig(
        type=LensType.LOGIT_LENS,
        source=ModelLocatorConfig(
            activation=ModelLayerLocation.RESIDUAL_POST,
            location=[ModelLocation.LAYER, ModelLocation.POSITION],
        ),
    )


def _coordinates_config() -> LensConfig:
    return LensConfig(
        type=LensType.LOGIT_LENS,
        source=ModelLocatorConfig(
            coordinates=[
                {"activation": "resid.post", "layer": -2, "position": -1},
                {"activation": "mlp.out", "layer": -1, "position": -1},
            ]
        ),
    )


def test_prepare_populates_input(model):
    lens_input = LensInput(PROMPT, _declarative_config()).prepare(model)

    assert lens_input.tokens.shape[0] == 1
    assert lens_input.logits.shape[:2] == lens_input.tokens.shape
    assert lens_input.cache["blocks.0.hook_resid_post"].shape[1] == lens_input.tokens.shape[1]
    assert len(lens_input.index) == model.cfg.n_layers * lens_input.tokens.shape[1]


def test_cache_is_restricted_to_used_activations(model):
    lens_input = LensInput(PROMPT, _declarative_config()).prepare(model)

    # resid.post over every layer -> exactly one hook per layer and nothing else is cached.
    expected = {f"blocks.{layer}.hook_resid_post" for layer in range(model.cfg.n_layers)}
    assert lens_input.index.source_activation_names() == expected
    assert set(lens_input.cache.keys()) == expected

    # ...and that is a strict subset of what an unfiltered run would have cached.
    _, full_cache = model.run_with_cache(lens_input.tokens)
    assert set(lens_input.cache.keys()) < set(full_cache.keys())


def test_prepare_is_idempotent(model):
    lens_input = LensInput(PROMPT, _declarative_config()).prepare(model)
    index = lens_input.index
    assert lens_input.prepare(model).index is index


def test_unprepared_access_asserts():
    with pytest.raises(AssertionError):
        _ = LensInput(PROMPT, _declarative_config()).tokens


def test_declarative_index_covers_full_grid(model):
    lens_input = LensInput(PROMPT, _declarative_config()).prepare(model)
    index = lens_input.index
    n_pos = lens_input.tokens.shape[1]

    assert index.axis_range == [model.cfg.n_layers, n_pos]
    assert index.prepare_output_placeholder().shape == torch.Size([model.cfg.n_layers, n_pos])
    assert {el.output for el in index} == {(layer, pos) for layer in range(model.cfg.n_layers) for pos in range(n_pos)}


def test_elements_slice_the_cache(model):
    lens_input = LensInput(PROMPT, _declarative_config()).prepare(model)

    for element in lens_input.index:
        layer = element.source[ModelLocation.LAYER]
        position = element.source[ModelLocation.POSITION]
        activation_name = element.source.transformer_activation()

        assert activation_name == f"blocks.{layer}.hook_resid_post"
        assert torch.equal(
            lens_input.cache[activation_name][element.source.slices()],
            lens_input.cache[activation_name][:, position, :],
        )


def test_coordinates_index(model):
    lens_input = LensInput(PROMPT, _coordinates_config()).prepare(model)
    index = lens_input.index

    assert index.axis_range == [2]
    assert index.output_df.columns.to_list() == ["coordinate"]

    elements = list(index)
    assert [el.output for el in elements] == [(0,), (1,)]
    assert elements[0].source[ModelLocation.LAYER] == model.cfg.n_layers - 2
    assert elements[0].source.transformer_activation() == f"blocks.{model.cfg.n_layers - 2}.hook_resid_post"
    assert elements[1].source.transformer_activation() == f"blocks.{model.cfg.n_layers - 1}.hook_mlp_out"


def test_to_csv(model, tmp_path):
    lens_input = LensInput(PROMPT, _declarative_config()).prepare(model)
    path = tmp_path / "index.csv"
    lens_input.index.to_csv(str(path))

    df = pd.read_csv(path)
    assert len(df) == len(lens_input.index)
    assert all(col.startswith("source_") or col.startswith("index_") for col in df.columns)
    assert sum(col.startswith("source_") for col in df.columns) == 3  # layer, position, activation
    assert sum(col.startswith("index_") for col in df.columns) == 2  # layer, position
