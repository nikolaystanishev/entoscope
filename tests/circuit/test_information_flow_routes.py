"""Information Flow Routes, read off the circuits a discovery run returns."""

from collections import Counter

import networkx as nx
import pytest
import torch

from entoscope.circuit.collector import CircuitCollector
from entoscope.circuit.config import CircuitConfig
from entoscope.circuit.discoverer import CircuitDiscoverer
from entoscope.circuit.input import CircuitInput
from entoscope.circuit.method.circuit import PromptCircuit
from entoscope.circuit.method.factory import CircuitMethodFactory
from entoscope.circuit.method.information_flow_routes import InformationFlowRoutesConfig
from entoscope.model.cache import CachedModel
from entoscope.model.weight import ModelWeight, model_weight

PROMPT = "The capital of France is"


def circuit_config(**overrides) -> CircuitConfig:
    return CircuitConfig(
        config_id="ifr", method=InformationFlowRoutesConfig(type="information_flow_routes", **overrides)
    )


def discover(model, prompts=(PROMPT,), config: CircuitConfig | None = None) -> list[PromptCircuit]:
    collector = CircuitDiscoverer(model, collector_class=CircuitCollector).apply(
        CircuitInput(config or circuit_config(), list(prompts)), return_collector=True, verbose=False
    )
    assert collector is not None
    return collector.result


@pytest.fixture(scope="module")
def circuit(model) -> PromptCircuit:
    return discover(model)[0]


def test_the_graph_carries_a_node_per_component_and_token(model, circuit):
    n_tokens = model.to_tokens(PROMPT).shape[1]

    assert isinstance(circuit.graph, nx.MultiDiGraph)
    assert isinstance(circuit.routes, nx.MultiDiGraph)
    assert circuit.str_tokens == model.to_str_tokens(model.to_tokens(PROMPT)[0])

    kinds = Counter(attributes["kind"] for _, attributes in circuit.graph.nodes(data=True))
    assert kinds["input"] == n_tokens
    assert kinds["after_attn"] == model.cfg.n_layers * n_tokens
    assert kinds["mlp"] == model.cfg.n_layers * n_tokens
    assert kinds["after_ffn"] == model.cfg.n_layers * n_tokens


def test_attention_edges_are_kept_apart_per_head(model, circuit):
    attention = [(u, v, d) for u, v, d in circuit.graph.edges(data=True) if d["kind"] == "attention"]

    assert attention, "expected some attention edges above the threshold"
    assert min(d["head"] for _, _, d in attention) >= 0
    assert max(d["head"] for _, _, d in attention) < model.cfg.n_heads
    # One (source, target) pair carrying more than one head proves the parallel per-head edges a
    # plain DiGraph - which would sum over heads - could not represent.
    assert max(Counter((u, v) for u, v, _ in attention).values()) > 1


def test_the_kept_contributions_of_a_representation_renormalize_to_one(circuit):
    # Each attention head (or the MLP) contributes its share of what the representation becomes; the
    # ones left above the threshold are rescaled so a representation is still fully accounted for.
    for node, attributes in circuit.graph.nodes(data=True):
        if attributes["kind"] not in ("after_attn", "after_ffn"):
            continue
        incoming = sum(d["weight"] for _, _, d in circuit.graph.in_edges(node, data=True))
        assert incoming == pytest.approx(1.0, abs=1e-3), f"{node} incoming sum {incoming}"

    weights = [d["weight"] for _, _, d in circuit.graph.edges(data=True)]
    assert weights
    assert all(0.0 < weight <= 1.0 + 1e-6 for weight in weights)


def test_the_routes_are_a_pruned_subgraph_that_reaches_the_prediction(model, circuit):
    last = model.to_tokens(PROMPT).shape[1] - 1
    output_node = f"I{model.cfg.n_layers - 1}_{last}"

    assert circuit.routes.number_of_edges() > 0
    assert output_node in circuit.routes
    full_edges = set(circuit.graph.edges(keys=True))
    assert all(edge in full_edges for edge in circuit.routes.edges(keys=True))
    inputs = [node for node, attributes in circuit.routes.nodes(data=True) if attributes["kind"] == "input"]
    assert any(nx.has_path(circuit.routes, source, output_node) for source in inputs)


def test_a_higher_threshold_prunes_more(model):
    sparse = discover(model, config=circuit_config(threshold=0.2))[0]
    dense = discover(model, config=circuit_config(threshold=0.0))[0]

    assert sparse.graph.number_of_edges() < dense.graph.number_of_edges()


def test_the_path_threshold_prunes_the_routes_on_its_own(model):
    shared = discover(model, config=circuit_config(threshold=0.02))[0]
    stricter_paths = discover(model, config=circuit_config(threshold=0.02, path_threshold=0.3))[0]

    # Both graphs are built with the same threshold, so only the routes may differ.
    assert stricter_paths.graph.number_of_edges() == shared.graph.number_of_edges()
    assert stricter_paths.routes.number_of_edges() < shared.routes.number_of_edges()


def test_the_traced_prediction_positions_are_configurable(model):
    traced_node = f"I{model.cfg.n_layers - 1}_1"

    last_only = discover(model, config=circuit_config(threshold=0.1))[0]
    two_positions = discover(model, config=circuit_config(threshold=0.1, prediction_positions=[-1, 1]))[0]

    assert traced_node not in last_only.routes
    assert traced_node in two_positions.routes
    assert f"I{model.cfg.n_layers - 1}_{model.to_tokens(PROMPT).shape[1] - 1}" in two_positions.routes


def test_one_circuit_is_built_per_prompt(model):
    prompts = [PROMPT, "The Eiffel Tower is in"]

    circuits = discover(model, prompts)

    assert [circuit.prompt for circuit in circuits] == prompts
    assert all(circuit.graph.number_of_edges() > 0 for circuit in circuits)


def test_a_run_through_a_cached_model_builds_the_same_graphs(model, tmp_path):
    # Circuit discovery caches an unfiltered forward pass per prompt.
    prompts = [PROMPT, "The Eiffel Tower is in"]
    cached_model = CachedModel(model, str(tmp_path))

    cached = discover(cached_model, prompts)
    plain = discover(model, prompts)

    assert len(list(tmp_path.rglob("*.pt"))) == len(prompts)
    for from_cache, direct in zip(cached, plain, strict=True):
        assert nx.utils.graphs_equal(from_cache.graph, direct.graph)
        assert nx.utils.graphs_equal(from_cache.routes, direct.routes)


def test_a_run_without_prompts_raises(model):
    with pytest.raises(ValueError, match="At least one prompt"):
        CircuitDiscoverer(model).apply(CircuitInput(circuit_config(), []), verbose=False)


def test_a_prediction_position_outside_the_prompt_raises(model):
    n_tokens = model.to_tokens(PROMPT).shape[1]

    with pytest.raises(ValueError, match=f"out of range for a {n_tokens}-token prompt"):
        discover(model, config=circuit_config(prediction_positions=[n_tokens]))


def test_a_threshold_that_keeps_no_edge_leaves_the_routes_empty(model):
    # Every contribution is a share of one, so nothing survives a threshold of 1.
    circuit = discover(model, config=circuit_config(threshold=1.0))[0]

    assert circuit.graph.number_of_edges() == 0
    assert circuit.routes.number_of_edges() == 0
    assert circuit.graph.number_of_nodes() > 0  # the representations are still there, unconnected


def test_a_circuit_without_edges_is_still_rendered(model, tmp_path):
    # The visualization collector has nothing to draw and says so instead of failing.
    save_dir = tmp_path / "empty_circuit"
    CircuitDiscoverer(model).apply(
        CircuitInput(circuit_config(threshold=1.0), [PROMPT]), save_dir=str(save_dir), verbose=False
    )

    assert "No edges above threshold" in (save_dir / "circuit_0.html").read_text()


def _head_outputs_by_key(model, cache, layer: int) -> torch.Tensor:
    """What each head writes to each query position from each key position, ``[query, key, head, d_model]``.

    Grouped-query attention shares a value head across a group of query heads, so the value heads are
    repeated to line up with the pattern's.
    """
    values = cache[f"blocks.{layer}.attn.hook_v"][0].double()
    values = values.repeat_interleave(model.cfg.n_heads // values.shape[1], dim=1)
    pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0].double()
    w_o = model_weight(model, ModelWeight.ATTENTION_OUT, layer).detach().double()
    return torch.einsum("hqk,khd,hdm->qkhm", pattern, values, w_o)


def _shares(parts: torch.Tensor, whole: torch.Tensor) -> torch.Tensor:
    """Ferrando and Voita's proximity of each part to the whole, as a share of all the parts' proximity.

    A part's proximity is how much nearer than nothing it puts the whole, in L1, floored at a small
    epsilon so a part pointing away still counts for a little. The distance carries the 1e-6 that torch's
    ``pairwise_distance`` adds to a difference, as the reference implementation measures it.
    """
    proximity = (whole.abs().sum() - (parts - whole + 1e-6).abs().sum(dim=-1)).clip(min=1e-5)
    return proximity / proximity.sum()


# A share is the small difference of two L1 norms that GPT-2's residual stream puts in the thousands,
# so the run's float32 rounding moves it by up to ~7e-5; the expectation is computed in float64.
SHARE = {"abs": 2e-4}


@pytest.mark.parametrize("model_fixture", ["model", "gqa_model"])
def test_every_edge_carries_its_parts_share_of_the_representation(request, backend, model_fixture):
    # With no threshold nothing is pruned or renormalized, so every edge into a representation weighs
    # its part's share of it - the head outputs from every key and the residual stream for the state
    # after attention, the MLP output and the residual stream for the state after the block.
    model = request.getfixturevalue(model_fixture)
    circuit = discover(model, config=circuit_config(threshold=0.0))[0]
    with torch.no_grad():
        _, cache = model.run_with_cache(model.to_tokens(PROMPT))
    n_tokens = model.to_tokens(PROMPT).shape[1]

    for layer in range(model.cfg.n_layers):
        written = _head_outputs_by_key(model, cache, layer)
        # The split adds back up to what attention wrote, bias aside: the parts are the model's own.
        b_O = model.blocks[layer].attn.b_O
        attn_out = cache[f"blocks.{layer}.hook_attn_out"][0]
        assert torch.allclose(written.sum(dim=(1, 2)).float(), attn_out if b_O is None else attn_out - b_O, atol=1e-4)

        for token in range(n_tokens):
            resid_pre, resid_mid, resid_post, mlp_out = (
                cache[f"blocks.{layer}.{hook}"][0, token].double()
                for hook in ("hook_resid_pre", "hook_resid_mid", "hook_resid_post", "hook_mlp_out")
            )
            attention = _shares(torch.cat([written[token].flatten(0, 1), resid_pre[None]]), resid_mid)
            attention_by_part, residual = attention[:-1].reshape(n_tokens, model.cfg.n_heads), attention[-1]

            edges = {
                (circuit.graph.nodes[source]["position"], data["head"]): data["weight"]
                for source, _, data in circuit.graph.in_edges(f"A{layer}_{token}", data=True)
                if data["kind"] == "attention"
            }
            # One edge per key and per query head: the heads that share a value head stay apart.
            assert set(edges) == {(key, head) for key in range(n_tokens) for head in range(model.cfg.n_heads)}
            for (key, head), weight in edges.items():
                assert weight == pytest.approx(float(attention_by_part[key, head]), **SHARE), (layer, token, key, head)
            (residual_edge,) = [
                data["weight"]
                for _, _, data in circuit.graph.in_edges(f"A{layer}_{token}", data=True)
                if data["kind"] == "residual"
            ]
            assert residual_edge == pytest.approx(float(residual), **SHARE)

            mlp, residual = _shares(torch.stack([mlp_out, resid_mid]), resid_post)
            assert circuit.graph.edges[f"A{layer}_{token}", f"M{layer}_{token}", "mlp"]["weight"] == pytest.approx(
                float(mlp), **SHARE
            )
            assert circuit.graph.edges[f"A{layer}_{token}", f"I{layer}_{token}", "residual"]["weight"] == pytest.approx(
                float(residual), **SHARE
            )


def test_parallel_attention_and_mlp_is_rejected(parallel_model):
    # A parallel architecture never fires hook_resid_mid and decomposes the residual differently, so
    # the sequential flow this method builds does not apply to it.
    with pytest.raises(NotImplementedError):
        CircuitDiscoverer(parallel_model).apply(CircuitInput(circuit_config(), [PROMPT]), verbose=False)


# --------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------
def test_a_yaml_config_runs_with_its_defaults(model, config_from_yaml):
    config = config_from_yaml(CircuitConfig, "my_circuit", {"method": "information_flow_routes"})
    method_config = CircuitMethodFactory.create(model, config).method_config

    assert method_config == InformationFlowRoutesConfig(
        type="information_flow_routes", threshold=0.04, path_threshold=None, prediction_positions=[-1]
    )
    assert method_config.resolved_path_threshold() == 0.04  # the routes fall back to the graph's threshold
    assert discover(model, config=config)[0].graph.number_of_edges() > 0


def test_a_yaml_config_carries_its_explicit_values_into_the_run(model, config_from_yaml):
    method = {
        "type": "information_flow_routes",
        "threshold": 0.1,
        "path_threshold": 0.2,
        "prediction_positions": [-1, 0],
    }
    config = config_from_yaml(CircuitConfig, "my_circuit", {"method": method})

    assert CircuitMethodFactory.create(model, config).method_config.resolved_path_threshold() == 0.2
    loaded = discover(model, config=config)[0]
    without_path_pruning = discover(model, config=circuit_config(threshold=0.1, prediction_positions=[-1, 0]))[0]

    assert loaded.graph.number_of_edges() == without_path_pruning.graph.number_of_edges()
    assert loaded.routes.number_of_edges() < without_path_pruning.routes.number_of_edges()


def test_a_whole_number_threshold_is_taken_as_it_is_written(model, config_from_yaml):
    # YAML has no float literal here: the 1 lands in a float field unconverted, and the run uses it.
    raw = {"method": {"type": "information_flow_routes", "threshold": 1}}
    config = config_from_yaml(CircuitConfig, "my_circuit", raw)

    assert CircuitMethodFactory.create(model, config).method_config.threshold == 1
    assert discover(model, config=config)[0].graph.number_of_edges() == 0


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({}, "Missing required keys for CircuitConfig.*method"),
        ({"method": "information_flow_routes", "threshold": 0.04}, "Unknown keys for CircuitConfig.*threshold"),
        ({"method": "acdc"}, "Unknown circuit method: acdc"),
        (
            {"method": {"type": "information_flow_routes", "bogus": 1}},
            r"Unknown keys for InformationFlowRoutesConfig: \['bogus'\]",
        ),
    ],
)
def test_a_malformed_circuit_config_is_rejected_when_it_is_loaded(config_from_yaml, raw, message):
    with pytest.raises(ValueError, match=message):
        config_from_yaml(CircuitConfig, "broken", raw)
