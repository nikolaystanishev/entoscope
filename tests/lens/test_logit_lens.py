"""The logit lens, run end to end: every row's top-k is checked against the projection it names."""

import json
import subprocess
import sys

import pandas as pd
import pytest
import torch
import yaml

from intex.lens.collector import LensCollector
from intex.lens.config import LensConfig, LensType
from intex.lens.input import LensInput
from intex.lens.lens import Lens
from intex.lens.method.logit_lens import _attention_scale
from intex.model.cache import CachedModel
from intex.model.config import Filter, ModelLayerLocation, ModelLocation, ModelLocatorConfig
from intex.operation.config import ConfigFactory

from ..conftest import RenderedFigure, forbid_model_runs, rendered_figure
from .conftest import (
    HEAD,
    NEURON,
    PROMPT,
    assert_row_reads,
    coordinates_config,
    declarative_config,
    lens_rows,
    mlp_matrix,
    run_lens,
    single_row,
    unembed,
)

LAYER_POSITION_GRID = dict(
    activation=ModelLayerLocation.RESIDUAL_POST, location=[ModelLocation.LAYER, ModelLocation.POSITION]
)

LENS_YAML = {
    "my_lens": {
        "type": "logit_lens",
        "source": {"activation": "resid.post", "location": ["layer", "position"]},
        "top_k": 2,
    }
}


def _result_columns(result: pd.DataFrame) -> list[str]:
    return [column for column in result.columns if not column.startswith(("source_", "index_"))]


# --------------------------------------------------------------------------------------------------
# The residual stream — the lens is the model's own read-out applied to an earlier state
# --------------------------------------------------------------------------------------------------
def test_the_grid_covers_every_layer_and_position(model):
    result = lens_rows(model, declarative_config(**LAYER_POSITION_GRID))

    n_positions = model.to_tokens(PROMPT).shape[1]
    assert len(result) == model.cfg.n_layers * n_positions
    assert _result_columns(result) == ["token_0", "str_token_0", "probability_0"]
    assert sorted(result["source_layer"].unique()) == list(range(model.cfg.n_layers))
    assert sorted(result["source_position"].unique()) == list(range(n_positions))


def test_the_last_layer_reads_out_the_models_own_prediction(model, activations):
    logits, _ = activations
    result = lens_rows(model, declarative_config(**LAYER_POSITION_GRID))

    for _, row in result[result.source_layer == model.cfg.n_layers - 1].iterrows():
        assert_row_reads(model, row, logits[0, row["source_position"]])


def test_every_row_unembeds_the_residual_state_its_coordinate_names(model, activations):
    _, cache = activations
    result = lens_rows(model, declarative_config(**LAYER_POSITION_GRID))

    for _, row in result.iterrows():
        residual = cache[f"blocks.{row['source_layer']}.hook_resid_post"][:, row["source_position"]]
        assert_row_reads(model, row, unembed(model, residual))


def test_top_k_columns_are_ordered_by_probability(model, activations):
    _, cache = activations
    result = lens_rows(model, declarative_config(**LAYER_POSITION_GRID, top_k=3))

    assert _result_columns(result) == [
        f"{name}_{i}" for i in range(3) for name in ("token", "str_token", "probability")
    ]
    for _, row in result.iloc[[0, len(result) - 1]].iterrows():
        residual = cache[f"blocks.{row['source_layer']}.hook_resid_post"][:, row["source_position"]]
        assert_row_reads(model, row, unembed(model, residual), top_k=3)
        assert row["probability_0"] > row["probability_1"] > row["probability_2"]


def test_a_coordinate_run_reads_each_coordinates_own_activation(model, activations):
    _, cache = activations
    last = model.cfg.n_layers - 1
    result = lens_rows(
        model,
        coordinates_config(
            {"activation": "resid.post", "layer": -2, "position": -1},
            {"activation": "mlp.out", "layer": -1, "position": -1},
        ),
    )

    assert len(result) == 2
    assert_row_reads(model, result.iloc[0], unembed(model, cache[f"blocks.{last - 1}.hook_resid_post"][:, -1]))
    assert_row_reads(model, result.iloc[1], unembed(model, cache[f"blocks.{last}.hook_mlp_out"][:, -1]))


def test_the_prepared_input_carries_the_models_own_logits(model, activations):
    logits, _ = activations
    lens_input = LensInput(declarative_config(**LAYER_POSITION_GRID), PROMPT).prepare(model)

    assert torch.equal(lens_input.logits, logits)
    assert lens_input.tokens.shape == logits.shape[:2]


def test_preparing_an_input_twice_keeps_what_it_already_built(model):
    lens_input = LensInput(declarative_config(**LAYER_POSITION_GRID), PROMPT).prepare(model)
    index, cache = lens_input.index, lens_input.cache

    assert lens_input.prepare(model).index is index
    assert lens_input.cache is cache


def test_a_prepared_input_moves_to_a_device(model):
    lens_input = LensInput(declarative_config(**LAYER_POSITION_GRID), PROMPT).prepare(model)

    moved = lens_input.to("cpu")

    assert moved is lens_input
    assert moved.tokens.device.type == "cpu"
    assert moved.logits.device.type == "cpu"
    assert all(activation.device.type == "cpu" for activation in moved.cache.values())


def test_the_run_only_caches_the_activations_its_index_names(model):
    # Not visible in the result: the restriction is about what the forward pass keeps, so the prepared
    # input is read directly. Nothing is stubbed — this is the real model and the real index.
    lens_input = LensInput(declarative_config(**LAYER_POSITION_GRID), PROMPT).prepare(model)

    expected = {f"blocks.{layer}.hook_resid_post" for layer in range(model.cfg.n_layers)}
    assert lens_input.index.source_activation_names() == expected
    # A bridge fires the block's own output hook under a second name for the same activation, so the
    # cache holds those aliases as well — and nothing beyond them.
    aliases = {f"blocks.{layer}.hook_out" for layer in range(model.cfg.n_layers)}
    assert expected <= set(lens_input.cache.keys()) <= expected | aliases

    _, full_cache = model.run_with_cache(lens_input.tokens)
    assert set(lens_input.cache.keys()) < set(full_cache.keys())


# --------------------------------------------------------------------------------------------------
# MLP-hidden activations — projected through the layer's W_out, so a row reads what its neurons write
# --------------------------------------------------------------------------------------------------
def test_a_single_neuron_is_lifted_by_its_row_of_w_out(model, activations):
    _, cache = activations
    layer = model.cfg.n_layers - 1
    row = single_row(
        model, coordinates_config({"activation": "mlp.post", "layer": -1, "position": -1, "neuron": NEURON})
    )

    activation = cache[f"blocks.{layer}.mlp.hook_post"][:, -1, NEURON]
    assert_row_reads(model, row, unembed(model, activation.unsqueeze(-1) * mlp_matrix(model, layer, "W_out")[NEURON]))


def test_a_list_of_neurons_sums_their_contributions(model, activations):
    _, cache = activations
    layer = model.cfg.n_layers - 1
    neurons = [NEURON, NEURON + 1]
    row = single_row(
        model, coordinates_config({"activation": "mlp.post", "layer": -1, "position": -1, "neuron": neurons})
    )

    w_out = mlp_matrix(model, layer, "W_out")
    activation = cache[f"blocks.{layer}.mlp.hook_post"][:, -1]
    contribution = sum(activation[:, neuron].unsqueeze(-1) * w_out[neuron] for neuron in neurons)
    assert_row_reads(model, row, unembed(model, contribution))


def test_every_neuron_together_is_mlp_out_without_its_bias(model, activations):
    _, cache = activations
    layer = model.cfg.n_layers - 1
    result = lens_rows(
        model,
        coordinates_config(
            {"activation": "mlp.post", "layer": -1, "position": -1},
            {"activation": "mlp.out", "layer": -1, "position": -1},
        ),
    )

    # W_out lifts every neuron's contribution; the output bias belongs to no neuron and is left out.
    without_bias = cache[f"blocks.{layer}.hook_mlp_out"][:, -1] - model.blocks[layer].mlp.b_out
    assert_row_reads(model, result.iloc[0], unembed(model, without_bias))
    # The second row is the same output *with* the bias, projected by the residual-stream lens.
    assert_row_reads(model, result.iloc[1], unembed(model, cache[f"blocks.{layer}.hook_mlp_out"][:, -1]))


def test_a_neuron_grid_projects_every_row_through_its_own_layer(model, activations):
    _, cache = activations
    neurons = [NEURON, NEURON + 1]
    config = declarative_config(
        ModelLayerLocation.MLP_POST,
        [ModelLocation.LAYER, ModelLocation.NEURON, ModelLocation.POSITION],
        filter=Filter.from_config([{"neuron": neurons, "position": -1}]),
    )
    result = lens_rows(model, config)

    assert len(result) == model.cfg.n_layers * len(neurons)
    assert sorted(result["source_neuron"].unique()) == neurons
    for _, row in result.iterrows():
        layer, neuron = row["source_layer"], row["source_neuron"]
        activation = cache[f"blocks.{layer}.mlp.hook_post"][:, -1, neuron]
        assert_row_reads(
            model, row, unembed(model, activation.unsqueeze(-1) * mlp_matrix(model, layer, "W_out")[neuron])
        )


# --------------------------------------------------------------------------------------------------
# Attention — head outputs are lifted by W_O, queries and keys are read against the embedding
# --------------------------------------------------------------------------------------------------
def test_a_single_head_is_lifted_by_its_row_of_w_o(model, activations):
    _, cache = activations
    layer = model.cfg.n_layers - 1
    row = single_row(model, coordinates_config({"activation": "attn.z", "layer": -1, "position": -1, "head": HEAD}))

    activation = cache[f"blocks.{layer}.attn.hook_z"][:, -1, HEAD]
    assert_row_reads(model, row, unembed(model, activation @ model.blocks[layer].attn.W_O[HEAD]))


def test_a_single_head_dimension_is_lifted_by_its_row_of_w_o(model, activations):
    _, cache = activations
    layer = model.cfg.n_layers - 1
    row = single_row(
        model, coordinates_config({"activation": "attn.z", "layer": -1, "position": -1, "head": HEAD, "neuron": 11})
    )

    activation = cache[f"blocks.{layer}.attn.hook_z"][:, -1, HEAD, 11]
    assert_row_reads(model, row, unembed(model, activation.unsqueeze(-1) * model.blocks[layer].attn.W_O[HEAD, 11]))


def test_a_list_of_heads_sums_their_contributions(model, activations):
    _, cache = activations
    layer = model.cfg.n_layers - 1
    heads = [HEAD, HEAD + 1]
    row = single_row(model, coordinates_config({"activation": "attn.z", "layer": -1, "position": -1, "head": heads}))

    w_o = model.blocks[layer].attn.W_O
    activation = cache[f"blocks.{layer}.attn.hook_z"][:, -1]
    contribution = sum(activation[:, head] @ w_o[head] for head in heads)
    assert_row_reads(model, row, unembed(model, contribution))


def test_every_head_together_is_attn_out_without_its_bias(model, activations):
    _, cache = activations
    layer = model.cfg.n_layers - 1
    row = single_row(model, coordinates_config({"activation": "attn.z", "layer": -1, "position": -1}))

    without_bias = cache[f"blocks.{layer}.hook_attn_out"][:, -1] - model.blocks[layer].attn.b_O
    assert_row_reads(model, row, unembed(model, without_bias))


def test_values_are_projected_through_the_same_route_as_the_head_output(model, activations):
    _, cache = activations
    layer = model.cfg.n_layers - 1
    row = single_row(model, coordinates_config({"activation": "attn.v", "layer": -1, "position": -1, "head": HEAD}))

    activation = cache[f"blocks.{layer}.attn.hook_v"][:, -1, HEAD]
    assert_row_reads(model, row, unembed(model, activation @ model.blocks[layer].attn.W_O[HEAD]))


@pytest.mark.parametrize(
    ("activation_name", "hook", "weight_name"),
    [("attn.q", "hook_q", "W_K"), ("attn.k", "hook_k", "W_Q")],
)
def test_queries_and_keys_are_read_against_the_embedding(model, activations, activation_name, hook, weight_name):
    # A query is scored against the keys of every vocabulary item, and a key against every query.
    _, cache = activations
    layer = model.cfg.n_layers - 1
    row = single_row(
        model, coordinates_config({"activation": activation_name, "layer": -1, "position": -1, "head": HEAD})
    )

    activation = cache[f"blocks.{layer}.attn.{hook}"][:, -1, HEAD]
    with torch.no_grad():
        weight = getattr(model.blocks[layer].attn, weight_name)[HEAD]
        expected = activation @ weight.T @ model.W_E.T / _attention_scale(model)
    assert_row_reads(model, row, expected)


def test_a_head_grid_projects_every_row_through_its_own_head(model, activations):
    _, cache = activations
    config = declarative_config(
        ModelLayerLocation.ATTENTION_Z,
        [ModelLocation.LAYER, ModelLocation.HEAD, ModelLocation.POSITION],
        filter=Filter.from_config([{"position": -1}]),
    )
    result = lens_rows(model, config)

    assert len(result) == model.cfg.n_layers * model.cfg.n_heads
    assert sorted(result["source_head"].unique()) == list(range(model.cfg.n_heads))
    for _, row in result.iloc[[0, len(result) - 1]].iterrows():
        layer, head = row["source_layer"], row["source_head"]
        activation = cache[f"blocks.{layer}.attn.hook_z"][:, -1, head]
        assert_row_reads(model, row, unembed(model, activation @ model.blocks[layer].attn.W_O[head]))


def test_a_run_through_a_cached_model_reads_the_same_activations(model, tmp_path):
    # The lens asks for a named subset of the hooks, so the cache stores exactly that one entry.
    cached_model = CachedModel(model, str(tmp_path))
    config = declarative_config(**LAYER_POSITION_GRID)

    first = lens_rows(cached_model, config)

    assert len(list(tmp_path.rglob("*.pt"))) == 1
    pd.testing.assert_frame_equal(first, lens_rows(model, config))
    pd.testing.assert_frame_equal(lens_rows(cached_model, config), first)  # now served from disk


# --------------------------------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------------------------------
def test_artifacts_are_saved_to_disk(model, config_from_yaml, tmp_path):
    config = config_from_yaml(LensConfig, "my_lens", LENS_YAML["my_lens"])

    save_dir = tmp_path / "lens_run"
    collector = run_lens(model, config, save_dir=save_dir)

    assert not (save_dir / "result.pt").exists()
    result_df = pd.read_csv(save_dir / "result.csv")
    assert len(result_df) == len(collector.result)
    assert result_df["token_0"].tolist() == collector.result["token_0"].tolist()
    assert result_df["str_token_1"].tolist() == collector.result["str_token_1"].tolist()
    assert result_df["probability_0"].tolist() == pytest.approx(collector.result["probability_0"].tolist())
    # The saved strings survive the CSV round-trip because the collector escapes them.
    assert all(token.isprintable() for token in result_df["str_token_0"] if isinstance(token, str))

    index_df = pd.read_csv(save_dir / "index.csv")
    assert len(index_df) == len(collector.operation_input.index)
    assert sum(column.startswith("source_") for column in index_df.columns) == 3  # layer, position, activation
    assert sum(column.startswith("index_") for column in index_df.columns) == 2  # layer, position

    with open(save_dir / "config.json", encoding="utf-8") as f:
        saved_config = json.load(f)
    assert saved_config["type"] == "logit_lens"
    assert saved_config["config_id"] == "my_lens"
    assert saved_config["top_k"] == 2


def test_a_run_without_a_collector_still_writes_its_artifacts(model, config_from_yaml, tmp_path):
    # Not asking for the collector is the default, and how the example scripts run an operation.
    config = config_from_yaml(LensConfig, "my_lens", LENS_YAML["my_lens"])
    expected = run_lens(model, config, save_dir=tmp_path / "with_collector").result

    save_dir = tmp_path / "without_collector"
    returned = Lens(model).apply(LensInput(config, PROMPT), save_dir=str(save_dir), verbose=False)

    # Nothing comes back, but the run did all of its work and left it on disk.
    assert returned is None
    pd.testing.assert_frame_equal(
        pd.read_csv(save_dir / "result.csv"), pd.read_csv(tmp_path / "with_collector" / "result.csv")
    )
    assert len(pd.read_csv(save_dir / "result.csv")) == len(expected)
    assert (save_dir / "index.csv").exists()


def test_a_yaml_config_runs_the_same_way_a_code_built_one_does(model, config_from_yaml):
    from_file = config_from_yaml(LensConfig, "my_lens", LENS_YAML["my_lens"])
    in_code = LensConfig(
        type=LensType.LOGIT_LENS,
        top_k=2,
        source=ModelLocatorConfig(
            activation=ModelLayerLocation.RESIDUAL_POST,
            location=[ModelLocation.LAYER, ModelLocation.POSITION],
        ),
    )

    assert lens_rows(model, from_file).equals(lens_rows(model, in_code))


def test_a_config_id_the_file_does_not_have_raises(tmp_path):
    path = tmp_path / "configs.yaml"
    path.write_text(yaml.safe_dump(LENS_YAML))
    factory = ConfigFactory[LensConfig](str(path))

    assert factory.create("my_lens").config_id == "my_lens"
    with pytest.raises(KeyError):
        factory.create("missing")


# --------------------------------------------------------------------------------------------------
# Driven from an experiment script — the way README and example_lens_script.py run a lens
# --------------------------------------------------------------------------------------------------
LENS_SCRIPT = """\
from dataclasses import dataclass

from transformer_lens import HookedTransformer, HookedTransformerConfig

from intex.lens.config import LensConfig
from intex.lens.input import LensInput
from intex.lens.lens import Lens
from intex.operation.config import ConfigFactory
from intex.script import experiment

DESCRIPTION = "Run a lens from a YAML config"


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
    collector = Lens(model).apply(
        LensInput(lens_config, "Paris is located in"), save_dir=config.save_dir, return_collector=True, verbose=False
    )
    print(len(collector.result))
"""


def test_the_lens_runs_from_an_experiment_script(tmp_path):
    # A fresh process, the config read off a file by its id, and the artifacts left behind.
    script = tmp_path / "lens_script.py"
    script.write_text(LENS_SCRIPT)
    config_path = tmp_path / "configs.yaml"
    config_path.write_text(yaml.safe_dump(LENS_YAML))
    save_dir = tmp_path / "lens_run"

    result = subprocess.run(
        [sys.executable, str(script), "--config-path", str(config_path)]
        + ["--config-id", "my_lens", "--save-dir", str(save_dir)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == str(len(pd.read_csv(save_dir / "result.csv")))
    assert (save_dir / "index.csv").exists()
    assert json.loads((save_dir / "config.json").read_text())["config_id"] == "my_lens"


# --------------------------------------------------------------------------------------------------
# Rejections
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"source": {"activation": "resid.post", "location": ["layer"]}}, "Missing required keys for LensConfig.*type"),
        ({"type": "logit_lens"}, "Missing required keys for LensConfig.*source"),
        ({"type": "attention_lens", "source": {"activation": "resid.post"}}, "attention_lens"),
        (
            {"type": "logit_lens", "source": {"activation": "resid.post"}, "target": {}},
            "Unknown keys for LensConfig.*target",
        ),
    ],
)
def test_a_malformed_lens_config_is_rejected_when_it_is_loaded(config_from_yaml, raw, message):
    with pytest.raises(ValueError, match=message):
        config_from_yaml(LensConfig, "broken", raw)


def test_a_lens_type_that_is_not_known_raises(model):
    # A config built in code can name a type the factory has no lens for.
    config = declarative_config(**LAYER_POSITION_GRID)
    config.type = "attention_lens"

    with pytest.raises(ValueError, match="Unknown lens type"):
        Lens(model).apply(LensInput(config, PROMPT), verbose=False)


def test_an_activation_without_a_vocabulary_projection_raises(model):
    config = coordinates_config({"activation": "attn.pattern", "layer": 0, "head": 0, "position": -1})

    with pytest.raises(ValueError, match="only residual-stream, MLP-hidden and attention head"):
        Lens(model).apply(LensInput(config, PROMPT), verbose=False)


@pytest.mark.parametrize("activation_name", ["attn.k", "attn.v"])
def test_key_and_value_projections_are_rejected_on_grouped_query_attention(gqa_model, activation_name):
    config = coordinates_config({"activation": activation_name, "layer": 0, "head": 0, "position": -1})

    with pytest.raises(ValueError, match="grouped-query attention"):
        Lens(gqa_model).apply(LensInput(config, PROMPT), verbose=False)


def test_key_and_value_projections_are_allowed_without_grouping(small_model):
    # The same activation on multi-head attention passes the check and projects like any head output.
    config = coordinates_config({"activation": "attn.v", "layer": 0, "head": 0, "position": -1})

    result = lens_rows(small_model, config)

    assert len(result) == 1
    assert 0 <= result.iloc[0]["token_0"] < small_model.cfg.d_vocab


def test_a_lens_config_without_a_prompt_raises():
    with pytest.raises(ValueError, match="'logit_lens' reads activations, so it needs a prompt"):
        LensInput(declarative_config(**LAYER_POSITION_GRID))


# --------------------------------------------------------------------------------------------------
# The plan a run would execute
# --------------------------------------------------------------------------------------------------
def test_a_plan_resolves_the_grid_without_running_the_model(model, monkeypatch, tmp_path):
    save_dir = tmp_path / "planned"
    lens_input = LensInput(declarative_config(**LAYER_POSITION_GRID), PROMPT)
    n_positions = model.to_tokens(PROMPT).shape[1]

    forbid_model_runs(monkeypatch, model)
    plan = Lens(model).plan(lens_input, save_dir=str(save_dir), verbose=False)

    assert not save_dir.exists()
    assert plan.steps == model.cfg.n_layers * n_positions
    assert plan.forward_passes == 1
    assert plan.prompts == [("prompt", PROMPT, n_positions)]
    # The names the run passes as its ``names_filter``, in the same sorted order.
    assert plan.activations == sorted(f"blocks.{layer}.hook_resid_post" for layer in range(model.cfg.n_layers))


def test_a_plan_describes_the_run_that_follows_it(model, config_from_yaml, tmp_path):
    config = config_from_yaml(LensConfig, "my_lens", LENS_YAML["my_lens"])
    lens_input = LensInput(config, PROMPT)
    save_dir = tmp_path / "lens_run"

    plan = Lens(model).plan(lens_input, save_dir=str(save_dir), verbose=False)
    # The same input object: planning resolves its index, the run still has to fetch activations.
    collector = Lens(model).apply(lens_input, save_dir=str(save_dir), return_collector=True, verbose=False)
    assert collector is not None

    assert len(collector.result) == plan.steps
    assert sorted(path.name for path in save_dir.iterdir()) == sorted(plan.outputs)
    pd.testing.assert_frame_equal(plan.index[0], collector.operation_input.index.to_df())

    index_df = pd.read_csv(save_dir / "index.csv")
    assert len(index_df) == len(plan.index[0])
    assert list(index_df.columns) == list(plan.index[0].columns)


def test_a_plan_prints_its_summary_and_its_index(model, config_from_yaml):
    config = config_from_yaml(LensConfig, "my_lens", LENS_YAML["my_lens"])

    described = str(Lens(model).plan(LensInput(config, PROMPT), verbose=False))

    assert described.splitlines()[0] == "Lens plan — logit_lens (config: my_lens)"
    assert f"{model.cfg.n_layers * model.to_tokens(PROMPT).shape[1]} elements" in described
    assert "1 forward pass" in described
    assert "writes: nothing (no save_dir)" in described
    # Locations read as they do in index.csv, not as the enum members the DataFrame holds.
    assert "resid.post" in described and "RESIDUAL_POST" not in described


def test_a_plan_prints_itself_unless_it_is_silenced(model, capsys):
    # Printing is the point of a dry run, so it is on by default; the plan comes back either way.
    lens_input = LensInput(declarative_config(**LAYER_POSITION_GRID), PROMPT)

    plan = Lens(model).plan(lens_input)

    assert capsys.readouterr().out == f"{plan}\n"
    assert Lens(model).plan(lens_input, verbose=False) is not None
    assert capsys.readouterr().out == ""


def test_a_long_index_is_truncated_when_it_is_printed(model):
    config = declarative_config(**LAYER_POSITION_GRID)

    described = Lens(model).plan(LensInput(config, PROMPT), verbose=False).describe(max_rows=4)

    assert len(described.splitlines()) < model.cfg.n_layers * model.to_tokens(PROMPT).shape[1]
    assert ".." in described


# --------------------------------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------------------------------
def _draw(model, config: LensConfig, save_dir, prompt: str | None = PROMPT) -> RenderedFigure:
    Lens(model).apply(LensInput(config, prompt), save_dir=str(save_dir), verbose=False)
    return rendered_figure(save_dir)


def test_a_lens_run_writes_an_interactive_result_page(model, tmp_path):
    save_dir = tmp_path / "lens_run"
    config = declarative_config(**LAYER_POSITION_GRID, top_k=3)
    expected = lens_rows(model, config)

    figure = _draw(model, config, save_dir)

    assert "Plotly.newPlot" in (save_dir / "result.html").read_text(encoding="utf-8")
    assert figure.title.startswith("Lens — logit_lens")
    assert figure.kind == "heatmap"
    # Probabilities are absolute, so the scale is pinned and two runs stay comparable at a glance.
    assert (figure.trace["zmin"], figure.trace["zmax"]) == (0.0, 1.0)
    assert figure.trace["colorbar"]["title"]["text"] == "probability"
    # The top-1 token is printed into its own cell, so it reaches the page as more than hover text.
    assert figure.trace["text"][0][0] == expected["str_token_0"].iloc[0]


def test_the_position_axis_is_labelled_with_the_prompts_tokens(model, tmp_path):
    figure = _draw(model, declarative_config(**LAYER_POSITION_GRID), tmp_path / "lens_run")

    assert (figure.axis_title("xaxis"), figure.axis_title("yaxis")) == ("position", "layer")
    assert figure.ticks("xaxis")[1] == '1: "The"'
    assert figure.ticks("yaxis") == [str(layer) for layer in range(model.cfg.n_layers)]


def test_a_head_axis_goes_across_with_the_layers_down(small_model, tmp_path):
    # The position is pinned to one value, so it carries no comparison and leaves the axes entirely.
    config = declarative_config(
        activation=ModelLayerLocation.ATTENTION_Z,
        location=[ModelLocation.LAYER, ModelLocation.HEAD, ModelLocation.POSITION],
        filter=Filter(inclusion_terms=[{ModelLocation.POSITION: -1}]),
    )

    figure = _draw(small_model, config, tmp_path / "heads")

    assert figure.kind == "heatmap"
    assert (figure.axis_title("xaxis"), figure.axis_title("yaxis")) == ("head", "layer")
    assert len(figure.trace["z"]) == small_model.cfg.n_layers
    assert len(figure.trace["z"][0]) == small_model.cfg.n_heads


def test_a_third_axis_folds_into_the_y_ticks(small_model, tmp_path):
    # Nothing is aggregated away: every (layer, head) pair keeps its own row of positions.
    config = declarative_config(
        activation=ModelLayerLocation.ATTENTION_Z,
        location=[ModelLocation.LAYER, ModelLocation.HEAD, ModelLocation.POSITION],
    )

    figure = _draw(small_model, config, tmp_path / "folded")

    assert figure.axis_title("xaxis") == "position"
    assert figure.axis_title("yaxis") == "layer · head"
    assert figure.ticks("yaxis")[0] == "layer 0 · head 0"
    assert len(figure.trace["z"]) == small_model.cfg.n_layers * small_model.cfg.n_heads


def test_the_base_collector_skips_the_html(model, tmp_path):
    # The base collector persists the CSVs only; visualization is opt-in via the subclass.
    save_dir = tmp_path / "no_viz"
    collector = Lens(model, collector_class=LensCollector).apply(
        LensInput(declarative_config(**LAYER_POSITION_GRID), PROMPT),
        save_dir=str(save_dir),
        return_collector=True,
        verbose=False,
    )

    assert collector is not None
    assert type(collector) is LensCollector
    assert (save_dir / "result.csv").exists()
    assert not (save_dir / "result.html").exists()


def test_a_plan_for_the_base_collector_promises_no_html(model, tmp_path):
    save_dir = tmp_path / "no_viz_plan"
    lens_input = LensInput(declarative_config(**LAYER_POSITION_GRID), PROMPT)
    lens = Lens(model, collector_class=LensCollector)

    plan = lens.plan(lens_input, save_dir=str(save_dir), verbose=False)
    lens.apply(lens_input, save_dir=str(save_dir), verbose=False)

    assert "result.html" not in plan.outputs
    assert sorted(path.name for path in save_dir.iterdir()) == sorted(plan.outputs)
