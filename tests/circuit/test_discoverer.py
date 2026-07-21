import json
from pathlib import Path

import networkx as nx
import yaml
from transformer_lens import HookedTransformer

from intex.circuit.collector import CircuitCollector, CircuitVisualizationCollector
from intex.circuit.config import CircuitConfig, CircuitMethodType
from intex.circuit.discoverer import CircuitDiscoverer
from intex.circuit.input import CircuitInput
from intex.circuit.method.circuit import PromptCircuit
from intex.operation.config import ConfigFactory

PROMPTS = ["The capital of France is", "The Eiffel Tower is in"]


def _config(**kwargs) -> CircuitConfig:
    return CircuitConfig(config_id="ifr", type=CircuitMethodType.INFORMATION_FLOW_ROUTES, **kwargs)


def test_one_circuit_per_prompt(model: HookedTransformer):
    collector = CircuitDiscoverer(model).apply(CircuitInput(PROMPTS, _config()), return_collector=True, verbose=False)
    assert collector is not None
    assert len(collector.result) == len(PROMPTS)
    assert all(isinstance(c, PromptCircuit) for c in collector.result)
    assert [c.prompt for c in collector.result] == PROMPTS
    assert all(c.graph.number_of_edges() > 0 for c in collector.result)


def test_artifacts_saved_to_disk(model: HookedTransformer, tmp_path: Path):
    # Build the config through ConfigFactory (YAML round-trip), mirroring how configs are sourced.
    config_yaml = tmp_path / "configs.yaml"
    config_yaml.write_text(yaml.safe_dump({"my_circuit": {"type": "information_flow_routes", "threshold": 0.05}}))
    config = ConfigFactory[CircuitConfig](str(config_yaml)).create("my_circuit")

    save_dir = tmp_path / "circuit_run"
    # CircuitDiscoverer defaults to the visualization collector, so HTML is written too.
    collector = CircuitDiscoverer(model).apply(
        CircuitInput(PROMPTS, config), save_dir=str(save_dir), return_collector=True, verbose=False
    )
    assert collector is not None
    assert isinstance(collector, CircuitVisualizationCollector)

    for i in range(len(PROMPTS)):
        assert (save_dir / f"graph_{i}.graphml").exists()
        assert (save_dir / f"graph_{i}.json").exists()
        assert (save_dir / f"routes_{i}.graphml").exists()
        assert (save_dir / f"routes_{i}.json").exists()
        assert (save_dir / f"circuit_{i}.html").exists()
    assert (save_dir / "prompts.json").exists()
    assert (save_dir / "config.json").exists()

    # GraphML round-trips to the same edge count and preserves per-head edge attributes.
    reloaded = nx.read_graphml(save_dir / "graph_0.graphml")
    assert isinstance(reloaded, nx.MultiDiGraph)
    assert reloaded.number_of_edges() == collector.result[0].graph.number_of_edges()
    assert any(d.get("kind") == "attention" for _, _, d in reloaded.edges(data=True))

    # node-link JSON round-trips too.
    data = json.loads((save_dir / "graph_0.json").read_text())
    from_json = nx.node_link_graph(data, edges="links")
    assert from_json.number_of_edges() == collector.result[0].graph.number_of_edges()

    # The HTML is self-contained (embeds plotly.js).
    html = (save_dir / "circuit_0.html").read_text()
    assert "Plotly.newPlot" in html or "plotly" in html.lower()

    prompts = json.loads((save_dir / "prompts.json").read_text())
    assert [p["prompt"] for p in prompts] == PROMPTS
    assert prompts[0]["route_edges"] <= prompts[0]["graph_edges"]

    saved_config = json.loads((save_dir / "config.json").read_text())
    assert saved_config["config_id"] == "my_circuit"
    assert saved_config["type"] == "information_flow_routes"
    assert saved_config["threshold"] == 0.05


def test_base_collector_skips_html(model: HookedTransformer, tmp_path: Path):
    # The base collector persists graph data only; visualization is opt-in via the subclass.
    save_dir = tmp_path / "no_viz"
    collector = CircuitDiscoverer(model, collector_class=CircuitCollector).apply(
        CircuitInput(PROMPTS[:1], _config()), save_dir=str(save_dir), return_collector=True, verbose=False
    )
    assert collector is not None
    assert type(collector) is CircuitCollector
    assert (save_dir / "graph_0.graphml").exists()
    assert (save_dir / "prompts.json").exists()
    assert not (save_dir / "circuit_0.html").exists()
