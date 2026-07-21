from collections import Counter

import networkx as nx
import pytest
import torch
from transformer_lens import HookedTransformer

from intex.circuit import activations
from intex.circuit.config import CircuitConfig, CircuitMethodType
from intex.circuit.method.information_flow_routes import InformationFlowRoutes

PROMPT = "The capital of France is"


def _config(threshold: float = 0.04, **kwargs) -> CircuitConfig:
    return CircuitConfig(type=CircuitMethodType.INFORMATION_FLOW_ROUTES, threshold=threshold, **kwargs)


def _build(model: HookedTransformer, threshold: float = 0.04, **kwargs):
    tokens = model.to_tokens(PROMPT)
    return InformationFlowRoutes(model, _config(threshold, **kwargs)).build(PROMPT, tokens), tokens


def test_graph_is_multidigraph_with_all_node_kinds(model: HookedTransformer):
    circuit, tokens = _build(model)
    n_tokens = tokens.shape[1]

    assert isinstance(circuit.graph, nx.MultiDiGraph)
    assert isinstance(circuit.routes, nx.MultiDiGraph)
    assert circuit.str_tokens == [t for t in model.to_str_tokens(tokens[0])]

    kinds = Counter(attrs["kind"] for _, attrs in circuit.graph.nodes(data=True))
    assert kinds["input"] == n_tokens
    assert kinds["after_attn"] == model.cfg.n_layers * n_tokens
    assert kinds["mlp"] == model.cfg.n_layers * n_tokens
    assert kinds["after_ffn"] == model.cfg.n_layers * n_tokens


def test_attention_edges_are_per_head(model: HookedTransformer):
    circuit, _ = _build(model)

    attn = [(u, v, d) for u, v, d in circuit.graph.edges(data=True) if d["kind"] == "attention"]
    assert attn, "expected some attention edges above threshold"
    heads = {d["head"] for _, _, d in attn}
    assert min(heads) >= 0
    assert max(heads) < model.cfg.n_heads

    # A single (source, target) pair carrying more than one head proves parallel per-head edges exist,
    # which a plain DiGraph (summing over heads) could not represent.
    pair_head_counts = Counter((u, v) for u, v, _ in attn)
    assert max(pair_head_counts.values()) > 1


def test_edge_weights_in_unit_interval(model: HookedTransformer):
    circuit, _ = _build(model)
    weights = [d["weight"] for _, _, d in circuit.graph.edges(data=True)]
    assert weights
    assert all(0.0 < w <= 1.0 + 1e-6 for w in weights)


def test_incoming_contributions_renormalize_to_one(model: HookedTransformer):
    # apply_threshold_and_renormalize guarantees each representation's kept incoming contributions
    # (attention heads + residual, or mlp + residual) sum to 1.
    circuit, _ = _build(model)

    for node, attrs in circuit.graph.nodes(data=True):
        if attrs["kind"] not in ("after_attn", "after_ffn"):
            continue
        incoming = sum(d["weight"] for _, _, d in circuit.graph.in_edges(node, data=True))
        assert incoming == pytest.approx(1.0, abs=1e-3), f"{node} incoming sum {incoming}"


def test_routes_reach_the_prediction(model: HookedTransformer):
    circuit, tokens = _build(model)
    n_layers = model.cfg.n_layers
    last = tokens.shape[1] - 1
    output_node = f"I{n_layers - 1}_{last}"

    assert circuit.routes.number_of_edges() > 0
    assert output_node in circuit.routes
    # Every route edge exists in the full graph (routes are a pruned subgraph).
    full_pairs = {(u, v) for u, v in circuit.graph.edges()}
    assert all((u, v) in full_pairs for u, v in circuit.routes.edges())
    # The output node is reachable from at least one input token through the routes.
    inputs = [n for n, a in circuit.routes.nodes(data=True) if a["kind"] == "input"]
    assert any(nx.has_path(circuit.routes, src, output_node) for src in inputs)


def test_higher_threshold_prunes_more(model: HookedTransformer):
    sparse, _ = _build(model, threshold=0.2)
    dense, _ = _build(model, threshold=0.0)
    assert sparse.graph.number_of_edges() <= dense.graph.number_of_edges()


def test_decomposed_attn_reconstructs_attn_out(model: HookedTransformer):
    # Summing the per-(key_pos, head) decomposition recovers the attention block output minus its bias.
    tokens = model.to_tokens(PROMPT)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    layer = model.cfg.n_layers // 2
    decomposed = activations.decomposed_attn(model, cache, layer)  # [pos, key_pos, head, d_model]
    reconstructed = decomposed.sum(dim=(1, 2))  # [pos, d_model]

    expected = cache["attn_out", layer][0] - model.blocks[layer].attn.b_O
    assert torch.allclose(reconstructed, expected, atol=1e-4)
