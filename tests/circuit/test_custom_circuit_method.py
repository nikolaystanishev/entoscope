"""A circuit-discovery method a user registers, run end to end the way the built-in one is."""

import json

import networkx as nx
import pytest

from entoscope.circuit.config import CircuitConfig
from entoscope.circuit.discoverer import CircuitDiscoverer
from entoscope.circuit.input import CircuitInput
from entoscope.circuit.method.circuit import AbstractCircuitMethod, PromptCircuit
from entoscope.circuit.method.factory import CircuitMethodFactory
from entoscope.model.token import str_token_labels

from ..conftest import rendered_figures, returned_figures

PROMPTS = ["The capital of France is", "The Eiffel Tower is in"]


class ResidualChain(AbstractCircuitMethod):
    """Every position's residual stream carried straight up the layers; the routes are the last one's."""

    def build(self, prompt, tokens):
        str_tokens = str_token_labels(self.model, tokens)
        graph = nx.MultiDiGraph()
        for position, token in enumerate(str_tokens):
            below = f"X0_{position}"
            graph.add_node(below, kind="input", layer=0, position=position, token=token)
            for layer in range(self.model.cfg.n_layers):
                node = f"I{layer}_{position}"
                graph.add_node(node, kind="after_ffn", layer=layer, position=position, token=token)
                graph.add_edge(below, node, key="residual", weight=1.0, kind="residual", head=-1)
                below = node

        routes = graph.subgraph(nx.ancestors(graph, below) | {below}).copy()
        return PromptCircuit(prompt=prompt, str_tokens=str_tokens, graph=graph, routes=routes)


@pytest.fixture
def residual_chain(registries):
    # It names no method config of its own, so it takes CircuitMethodConfig, which no field fits.
    return CircuitMethodFactory.register("residual_chain")(ResidualChain)


def test_a_registered_method_discovers_and_draws_a_circuit_per_prompt(
    small_model, config_from_yaml, residual_chain, tmp_path
):
    config = config_from_yaml(CircuitConfig, "chain", {"method": "residual_chain"})
    save_dir = tmp_path / "chain"

    collector = CircuitDiscoverer(small_model).apply(
        CircuitInput(config, PROMPTS), save_dir=str(save_dir), return_collector=True, verbose=False
    )

    n_layers = small_model.cfg.n_layers
    figures = collector.figures
    assert len(figures) == len(PROMPTS)
    for i, prompt in enumerate(PROMPTS):
        circuit = collector.result[i]
        assert circuit.graph.number_of_edges() == small_model.to_tokens(prompt).shape[1] * n_layers
        assert circuit.routes.number_of_edges() == n_layers
        assert nx.read_graphml(save_dir / f"graph_{i}.graphml").number_of_edges() == circuit.graph.number_of_edges()

        routes, graph = rendered_figures(save_dir, f"circuit_{i}.html")
        assert routes.title.startswith(f'Routes to the prediction · residual_chain · "{prompt}"')
        assert graph.title.startswith(f'Full contribution graph · residual_chain · "{prompt}"')
        assert returned_figures(figures[i]) == {"routes": routes, "graph": graph}
        page = (save_dir / f"circuit_{i}.html").read_text(encoding="utf-8")
        assert f"<title>Circuit · residual_chain · {prompt} (config: chain)</title>" in page

    saved = json.loads((save_dir / "config.json").read_text())
    assert saved["method"] == {"type": "residual_chain"}


def test_a_method_that_names_no_activations_plans_to_read_them_all(small_model, config_from_yaml, residual_chain):
    # ``activation_names`` left at its default says the method caches every activation, and a plan says so.
    config = config_from_yaml(CircuitConfig, "chain", {"method": "residual_chain"})

    plan = CircuitDiscoverer(small_model).plan(CircuitInput(config, PROMPTS), verbose=False)

    assert plan.reads_full_cache
    assert plan.activations == []
    assert "  reads: full activation cache" in str(plan).splitlines()
