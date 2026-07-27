"""The parameter lens, run end to end — it reads weights, so a run takes no prompt at all."""

import json

import pandas as pd
import pytest
import torch

from intex.lens.config import LensConfig, LensType, ProjectionSpace
from intex.lens.input import LensInput
from intex.lens.lens import Lens
from intex.model.config import Filter, ModelLayerLocation, ModelLocation, ModelLocatorConfig

from ..conftest import RenderedFigure, forbid_model_runs, rendered_figure
from .conftest import (
    HEAD,
    LAYER,
    NEURON,
    PROMPT,
    assert_row_reads,
    coordinates_config,
    lens_rows,
    mlp_matrix,
    run_lens,
    single_row,
)

PARAMETER_LENS_YAML = {
    "type": "parameter_lens",
    "top_k": 2,
    "source": {"activation": "mlp.post", "location": ["layer", "neuron"], "filter": [{"neuron": NEURON}]},
}


def _parameter_config(top_k: int = 1, projection: ProjectionSpace | None = None, scale: float = 1.0) -> LensConfig:
    return LensConfig(
        type=LensType.PARAMETER_LENS,
        top_k=top_k,
        projection=projection,
        scale=scale,
        source=ModelLocatorConfig(
            activation=ModelLayerLocation.MLP_POST,
            location=[ModelLocation.LAYER, ModelLocation.NEURON],
            filter=Filter(inclusion_terms=[{ModelLocation.NEURON: NEURON}]),
        ),
    )


def _weight_row(model, coordinate: dict, projection: ProjectionSpace | None = None, scale: float = 1.0) -> pd.Series:
    return single_row(
        model, coordinates_config(coordinate, lens_type=LensType.PARAMETER_LENS, projection=projection, scale=scale)
    )


def _projected(model, weight: torch.Tensor, projection: ProjectionSpace | None = None) -> torch.Tensor:
    with torch.no_grad():
        return weight @ (model.W_E.T if projection is ProjectionSpace.EMBEDDING else model.W_U)


# --------------------------------------------------------------------------------------------------
# A run without a prompt
# --------------------------------------------------------------------------------------------------
def test_a_parameter_lens_run_needs_no_prompt(model):
    result = lens_rows(model, _parameter_config())

    assert len(result) == model.cfg.n_layers
    assert result["source_neuron"].tolist() == [NEURON] * model.cfg.n_layers
    assert [column for column in result.columns if not column.startswith("source_")] == [
        "token_0",
        "str_token_0",
        "probability_0",
    ]


def test_a_parameter_lens_config_with_a_prompt_raises():
    with pytest.raises(ValueError, match="parameter lens .* takes no prompt"):
        LensInput(_parameter_config(), PROMPT)


# --------------------------------------------------------------------------------------------------
# One weight vector per row — the value vector a neuron writes, the key vector it reads with, and the
# subhead vectors of attention
# --------------------------------------------------------------------------------------------------
def test_an_mlp_value_vector_is_the_row_of_w_out(model):
    row = _weight_row(model, {"activation": "mlp.post", "layer": LAYER, "neuron": NEURON})

    assert_row_reads(model, row, _projected(model, mlp_matrix(model, LAYER, "W_out")[NEURON]))


def test_an_mlp_key_vector_is_the_column_of_w_in(model):
    row = _weight_row(model, {"activation": "mlp.pre", "layer": LAYER, "neuron": NEURON})

    assert_row_reads(model, row, _projected(model, mlp_matrix(model, LAYER, "W_in")[:, NEURON]))


@pytest.mark.parametrize(("activation_name", "weight_name"), [("attn.q", "W_Q"), ("attn.k", "W_K"), ("attn.v", "W_V")])
def test_an_attention_read_vector_is_the_column_of_its_weight(model, activation_name, weight_name):
    row = _weight_row(model, {"activation": activation_name, "layer": LAYER, "head": HEAD, "neuron": 12})

    weight = getattr(model.blocks[LAYER].attn, weight_name)
    assert_row_reads(model, row, _projected(model, weight[HEAD, :, 12]))


def test_an_attention_write_vector_is_the_row_of_w_o(model):
    row = _weight_row(model, {"activation": "attn.z", "layer": LAYER, "head": HEAD, "neuron": 12})

    assert_row_reads(model, row, _projected(model, model.blocks[LAYER].attn.W_O[HEAD, 12]))


def test_the_projection_space_switches_from_the_unembedding_to_the_embedding(model):
    weight = mlp_matrix(model, LAYER, "W_out")[NEURON]
    coordinate = {"activation": "mlp.post", "layer": LAYER, "neuron": NEURON}

    embedding = _weight_row(model, coordinate, projection=ProjectionSpace.EMBEDDING)
    unembedding = _weight_row(model, coordinate, projection=ProjectionSpace.UNEMBEDDING)

    assert_row_reads(model, embedding, _projected(model, weight, ProjectionSpace.EMBEDDING))
    assert_row_reads(model, unembedding, _projected(model, weight))
    # The two spaces score the vocabulary alike exactly when the model ties its embedding and its
    # unembedding — which GPT-2 does, until transformer_lens folds LayerNorm in and centers the
    # unembedding and the two come apart.
    tied = torch.equal(model.W_E.T, model.W_U)
    assert bool(embedding["probability_0"] == unembedding["probability_0"]) == tied


def test_a_negative_scale_flips_the_direction_the_weight_projects(model):
    weight = mlp_matrix(model, LAYER, "W_out")[NEURON]
    coordinate = {"activation": "mlp.post", "layer": LAYER, "neuron": NEURON}

    row = _weight_row(model, coordinate, scale=-1.0)

    # A negative scale inverts the direction, so the top token is the one the weight writes against.
    assert_row_reads(model, row, _projected(model, -weight))
    assert row["token_0"] != _weight_row(model, coordinate)["token_0"]


def test_a_positive_scale_sharpens_probabilities_without_changing_the_top_token(model):
    coordinate = {"activation": "mlp.post", "layer": LAYER, "neuron": NEURON}

    unscaled = _weight_row(model, coordinate)
    scaled = _weight_row(model, coordinate, scale=2.0)

    # Scaling before the softmax acts as a temperature: same argmax, sharper distribution.
    assert scaled["token_0"] == unscaled["token_0"]
    assert scaled["probability_0"] > unscaled["probability_0"]


# --------------------------------------------------------------------------------------------------
# Grouping — a list of neurons (or heads), and a location left out entirely, sum into one direction
# --------------------------------------------------------------------------------------------------
def test_grouped_neurons_sum_their_value_vectors_into_one_row(model):
    neurons = [NEURON, NEURON + 1]
    row = _weight_row(model, {"activation": "mlp.post", "layer": LAYER, "neuron": neurons})

    w_out = mlp_matrix(model, LAYER, "W_out")
    assert_row_reads(model, row, _projected(model, w_out[NEURON] + w_out[NEURON + 1]))


def test_grouped_neurons_sum_their_key_vectors_into_one_row(model):
    neurons = [NEURON, NEURON + 1]
    row = _weight_row(model, {"activation": "mlp.pre", "layer": LAYER, "neuron": neurons})

    w_in = mlp_matrix(model, LAYER, "W_in")
    assert_row_reads(model, row, _projected(model, w_in[:, NEURON] + w_in[:, NEURON + 1]))


def test_a_row_without_a_neuron_sums_every_neuron(model):
    row = _weight_row(model, {"activation": "mlp.post", "layer": LAYER})

    assert_row_reads(model, row, _projected(model, mlp_matrix(model, LAYER, "W_out").sum(0)))


def test_grouped_heads_and_neurons_sum_their_read_vectors(model):
    heads, neurons = [HEAD, HEAD + 1], [12, 13]
    row = _weight_row(model, {"activation": "attn.q", "layer": LAYER, "head": heads, "neuron": neurons})

    w_q = model.blocks[LAYER].attn.W_Q
    expected = sum(w_q[head, :, neuron] for head in heads for neuron in neurons)
    assert_row_reads(model, row, _projected(model, expected))


def test_grouped_heads_and_neurons_sum_their_write_vectors(model):
    heads, neurons = [HEAD, HEAD + 1], [12, 13]
    row = _weight_row(model, {"activation": "attn.z", "layer": LAYER, "head": heads, "neuron": neurons})

    w_o = model.blocks[LAYER].attn.W_O
    expected = sum(w_o[head, neuron] for head in heads for neuron in neurons)
    assert_row_reads(model, row, _projected(model, expected))


def test_a_row_without_a_head_sums_every_head(model):
    row = _weight_row(model, {"activation": "attn.z", "layer": LAYER, "neuron": 12})

    assert_row_reads(model, row, _projected(model, model.blocks[LAYER].attn.W_O[:, 12].sum(0)))


def test_top_k_tokens_come_from_the_projected_weight(model):
    result = lens_rows(model, _parameter_config(top_k=3))

    row = result[result.source_layer == LAYER].iloc[0]
    assert_row_reads(model, row, _projected(model, mlp_matrix(model, LAYER, "W_out")[NEURON]), top_k=3)


# --------------------------------------------------------------------------------------------------
# Gated MLPs — the key vector comes from the gate, and the linear branch has its own location
# --------------------------------------------------------------------------------------------------
def test_a_gated_mlp_reads_its_key_vector_from_the_gate(gated_model):
    row = _weight_row(gated_model, {"activation": "mlp.pre", "layer": 0, "neuron": 1})

    assert_row_reads(gated_model, row, _projected(gated_model, mlp_matrix(gated_model, 0, "W_gate")[:, 1]))


def test_a_gated_mlps_linear_branch_reads_w_in(gated_model):
    row = _weight_row(gated_model, {"activation": "mlp.pre_linear", "layer": 0, "neuron": 1})

    assert_row_reads(gated_model, row, _projected(gated_model, mlp_matrix(gated_model, 0, "W_in")[:, 1]))


def test_the_linear_branch_of_a_model_without_a_gate_raises(small_model):
    config = coordinates_config(
        {"activation": "mlp.pre_linear", "layer": 0, "neuron": 1}, lens_type=LensType.PARAMETER_LENS
    )

    with pytest.raises(ValueError, match="which this model does not have"):
        Lens(small_model).apply(LensInput(config), verbose=False)


# --------------------------------------------------------------------------------------------------
# Rejections
# --------------------------------------------------------------------------------------------------
def test_a_weight_vector_needs_a_single_layer(model):
    # Every weight lives in one block, so an element may not group layers.
    config = coordinates_config(
        {"activation": "mlp.post", "layer": [0, 1], "neuron": NEURON}, lens_type=LensType.PARAMETER_LENS
    )

    with pytest.raises(ValueError, match="needs a single layer"):
        Lens(model).apply(LensInput(config), verbose=False)


@pytest.mark.parametrize("activation_name", ["attn.k", "attn.v"])
def test_key_and_value_weights_are_rejected_on_grouped_query_attention(gqa_model, activation_name):
    config = coordinates_config(
        {"activation": activation_name, "layer": 0, "head": 0, "neuron": 1}, lens_type=LensType.PARAMETER_LENS
    )

    with pytest.raises(ValueError, match="grouped-query attention"):
        Lens(gqa_model).apply(LensInput(config), verbose=False)


@pytest.mark.parametrize(
    "activation_name", ["embed", "resid.pre", "resid.mid", "resid.post", "attn.out", "attn.pattern", "mlp.out"]
)
def test_an_activation_that_names_no_weight_vector_raises(model, activation_name):
    config = coordinates_config({"activation": activation_name, "layer": 0}, lens_type=LensType.PARAMETER_LENS)

    with pytest.raises(ValueError, match="Parameter lens cannot project"):
        Lens(model).apply(LensInput(config), verbose=False)


@pytest.mark.parametrize(
    "source",
    [
        # weights have no position axis, declaratively...
        {"activation": "mlp.post", "location": ["layer", "position"], "filter": [{"neuron": NEURON}]},
        # ...or as a coordinate
        {"coordinates": [{"activation": "mlp.post", "layer": -1, "neuron": NEURON, "position": -1}]},
    ],
)
def test_a_position_is_not_a_parameter_lens_location(config_from_yaml, source):
    with pytest.raises(ValueError, match="'position' is not a parameter lens location"):
        config_from_yaml(LensConfig, "positioned", {"type": "parameter_lens", "source": source})


def test_a_projection_on_another_lens_type_raises(config_from_yaml):
    raw = {"type": "logit_lens", "projection": "embedding", "source": {"activation": "resid.post"}}

    with pytest.raises(ValueError, match="'projection' is a parameter lens setting"):
        config_from_yaml(LensConfig, "projected", raw)


def test_a_scale_on_another_lens_type_raises(config_from_yaml):
    raw = {"type": "logit_lens", "scale": -1.0, "source": {"activation": "resid.post"}}

    with pytest.raises(ValueError, match="'scale' is a parameter lens setting"):
        config_from_yaml(LensConfig, "scaled", raw)


def test_an_unknown_projection_space_raises(config_from_yaml):
    with pytest.raises(ValueError):
        config_from_yaml(LensConfig, "projected", {**PARAMETER_LENS_YAML, "projection": "residual"})


# --------------------------------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------------------------------
def test_artifacts_are_saved_to_disk(model, config_from_yaml, tmp_path):
    config = config_from_yaml(LensConfig, "my_parameter_lens", PARAMETER_LENS_YAML)

    save_dir = tmp_path / "parameter_lens_run"
    collector = run_lens(model, config, save_dir=save_dir)

    result_df = pd.read_csv(save_dir / "result.csv")
    assert result_df["token_0"].tolist() == collector.result["token_0"].tolist()
    assert result_df["probability_1"].tolist() == pytest.approx(collector.result["probability_1"].tolist())

    index_df = pd.read_csv(save_dir / "index.csv")
    assert len(index_df) == len(collector.operation_input.index)

    with open(save_dir / "config.json", encoding="utf-8") as f:
        saved_config = json.load(f)
    assert saved_config["type"] == "parameter_lens"
    assert saved_config["config_id"] == "my_parameter_lens"
    assert saved_config["top_k"] == 2


# --------------------------------------------------------------------------------------------------
# The plan a run would execute
# --------------------------------------------------------------------------------------------------
def test_a_plan_reads_no_activations_and_runs_nothing(model, monkeypatch, tmp_path):
    # A run over weights takes no prompt, so there is nothing to cache and no pass to make.
    save_dir = tmp_path / "planned"
    lens_input = LensInput(_parameter_config())

    forbid_model_runs(monkeypatch, model)
    plan = Lens(model).plan(lens_input, save_dir=str(save_dir), verbose=False)

    assert not save_dir.exists()
    assert plan.forward_passes == 0
    assert plan.activations == []
    assert plan.prompts == []
    assert plan.steps == model.cfg.n_layers
    assert "reads:" not in str(plan)


def test_a_plan_describes_the_run_that_follows_it(model, tmp_path):
    save_dir = tmp_path / "parameter_lens_run"
    lens_input = LensInput(_parameter_config())

    plan = Lens(model).plan(lens_input, save_dir=str(save_dir), verbose=False)
    collector = Lens(model).apply(lens_input, save_dir=str(save_dir), return_collector=True, verbose=False)
    assert collector is not None

    assert len(collector.result) == plan.steps
    assert sorted(path.name for path in save_dir.iterdir()) == sorted(plan.outputs)
    pd.testing.assert_frame_equal(plan.index[0], collector.operation_input.index.to_df())


# --------------------------------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------------------------------
def _draw(model, config: LensConfig, save_dir) -> RenderedFigure:
    Lens(model).apply(LensInput(config), save_dir=str(save_dir), verbose=False)
    return rendered_figure(save_dir)


def test_a_parameter_lens_page_needs_no_prompt(model, tmp_path):
    # One neuron read across every layer varies over the layer alone, so it is drawn as a bar — and
    # with no prompt there are no tokens to name the axis with, so the layers label themselves.
    save_dir = tmp_path / "parameter_lens_run"
    expected = lens_rows(model, _parameter_config(top_k=3))

    figure = _draw(model, _parameter_config(top_k=3), save_dir)

    assert figure.title.startswith("Lens — parameter_lens")
    assert "weights (no prompt)" in figure.title
    assert figure.kind == "bar"
    assert figure.ticks("xaxis") == [str(layer) for layer in range(model.cfg.n_layers)]
    assert figure.trace["text"][0] == expected["str_token_0"].iloc[0]


def test_a_neuron_axis_goes_across_with_the_layers_down(small_model, tmp_path):
    config = LensConfig(
        type=LensType.PARAMETER_LENS,
        source=ModelLocatorConfig(
            activation=ModelLayerLocation.MLP_POST, location=[ModelLocation.LAYER, ModelLocation.NEURON]
        ),
    )

    figure = _draw(small_model, config, tmp_path / "neurons")

    assert figure.kind == "heatmap"
    assert (figure.axis_title("xaxis"), figure.axis_title("yaxis")) == ("neuron", "layer")
    assert figure.ticks("xaxis") == [str(neuron) for neuron in range(small_model.cfg.d_mlp)]


def test_an_empty_index_still_writes_a_page(model, tmp_path):
    # Including and excluding the same layer leaves nothing to run, and a result with no rows has no
    # value column either — the page says so rather than failing to render.
    config = LensConfig(
        type=LensType.PARAMETER_LENS,
        source=ModelLocatorConfig(
            activation=ModelLayerLocation.MLP_POST,
            location=[ModelLocation.LAYER],
            filter=Filter(inclusion_terms=[{ModelLocation.LAYER: 0}], exclusion_terms=[{ModelLocation.LAYER: 0}]),
        ),
    )
    save_dir = tmp_path / "empty"

    _draw(model, config, save_dir)

    assert len(pd.read_csv(save_dir / "result.csv")) == 0
    assert "No rows in the index" in (save_dir / "result.html").read_text(encoding="utf-8")
