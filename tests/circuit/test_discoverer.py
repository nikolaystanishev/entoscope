import json
from html import escape
from pathlib import Path

import networkx as nx
import yaml

from entoscope.circuit.collector import CircuitCollector, CircuitVisualizationCollector
from entoscope.circuit.config import CircuitConfig
from entoscope.circuit.discoverer import CircuitDiscoverer
from entoscope.circuit.input import CircuitInput
from entoscope.circuit.method.information_flow_routes import InformationFlowRoutesConfig
from entoscope.model import Model
from entoscope.operation.config import ConfigFactory

from ..conftest import forbid_model_runs, rendered_figures

PROMPTS = ["The capital of France is", "The Eiffel Tower is in"]


def _config(**kwargs) -> CircuitConfig:
    return CircuitConfig(config_id="ifr", method=InformationFlowRoutesConfig(type="information_flow_routes", **kwargs))


def test_artifacts_saved_to_disk(model: Model, tmp_path: Path):
    # Build the config through ConfigFactory (YAML round-trip), mirroring how configs are sourced.
    config_yaml = tmp_path / "configs.yaml"
    config_yaml.write_text(
        yaml.safe_dump({"my_circuit": {"method": {"type": "information_flow_routes", "threshold": 0.05}}})
    )
    config = ConfigFactory[CircuitConfig](str(config_yaml)).create("my_circuit")

    save_dir = tmp_path / "circuit_run"
    # CircuitDiscoverer defaults to the visualization collector, so HTML is written too.
    collector = CircuitDiscoverer(model).apply(
        CircuitInput(config, PROMPTS), save_dir=str(save_dir), return_collector=True, verbose=False
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

    # The HTML is self-contained (embeds plotly.js) and holds both of the circuit's panels.
    html = (save_dir / "circuit_0.html").read_text()
    assert "Plotly.newPlot" in html or "plotly" in html.lower()
    routes, graph = rendered_figures(save_dir, "circuit_0.html")
    assert routes.layout["title"]["text"].startswith("Routes to the prediction · information_flow_routes")
    assert graph.layout["title"]["text"].startswith("Full contribution graph")

    prompts = json.loads((save_dir / "prompts.json").read_text())
    assert [p["prompt"] for p in prompts] == PROMPTS
    assert prompts[0]["route_edges"] <= prompts[0]["graph_edges"]

    saved_config = json.loads((save_dir / "config.json").read_text())
    assert saved_config["config_id"] == "my_circuit"
    assert saved_config["method"] == {
        "type": "information_flow_routes",
        "threshold": 0.05,
        "path_threshold": None,
        "prediction_positions": [-1],
    }


def test_the_tokens_labelling_the_positions_are_escaped_for_the_page(model: Model, tmp_path: Path):
    # plotly reads a label as HTML, so a token such as Llama's BOS '<s>' would open a strikethrough and
    # vanish from the axis.
    prompt = "Fish <b>&chips</b>"
    save_dir = tmp_path / "markup"

    CircuitDiscoverer(model).apply(CircuitInput(_config(), [prompt]), save_dir=str(save_dir), verbose=False)

    _, graph = rendered_figures(save_dir, "circuit_0.html")
    tokens = model.to_str_tokens(prompt)
    assert graph.ticks("xaxis") == [f"{position}: {escape(token)}" for position, token in enumerate(tokens)]


def test_a_run_without_a_collector_still_writes_its_artifacts(model: Model, tmp_path: Path):
    # Not asking for the collector is the default, and how the example scripts run an operation.
    save_dir = tmp_path / "no_collector"
    returned = CircuitDiscoverer(model, collector_class=CircuitCollector).apply(
        CircuitInput(_config(), PROMPTS[:1]), save_dir=str(save_dir), verbose=False
    )

    # Nothing comes back, but the run did all of its work and left it on disk.
    assert returned is None
    assert (save_dir / "graph_0.graphml").exists()
    assert (save_dir / "routes_0.graphml").exists()
    assert json.loads((save_dir / "prompts.json").read_text())[0]["prompt"] == PROMPTS[0]


def test_a_plan_counts_one_graph_per_prompt_without_building_them(model: Model, monkeypatch, tmp_path):
    save_dir = tmp_path / "planned"
    circuit_input = CircuitInput(_config(), PROMPTS)

    forbid_model_runs(monkeypatch, model)
    plan = CircuitDiscoverer(model).plan(circuit_input, save_dir=str(save_dir), verbose=False)

    assert not save_dir.exists()
    assert plan.steps == len(PROMPTS)
    assert plan.forward_passes == len(PROMPTS)
    assert not plan.reads_full_cache
    assert len(plan.activations) == 6 * model.cfg.n_layers
    assert [prompt for _, prompt, _ in plan.prompts] == PROMPTS
    assert [n_tokens for _, _, n_tokens in plan.prompts] == [model.to_tokens(prompt).shape[1] for prompt in PROMPTS]


def test_a_plan_describes_the_run_that_follows_it(model: Model, tmp_path: Path):
    save_dir = tmp_path / "circuit_run"
    circuit_input = CircuitInput(_config(), PROMPTS[:1])

    plan = CircuitDiscoverer(model).plan(circuit_input, save_dir=str(save_dir), verbose=False)
    CircuitDiscoverer(model).apply(circuit_input, save_dir=str(save_dir), verbose=False)

    assert sorted(path.name for path in save_dir.iterdir()) == sorted(plan.outputs)
    assert "circuit_0.html" in plan.outputs


def test_the_base_collector_writes_what_its_plan_promises_without_the_page(model: Model, tmp_path: Path):
    save_dir = tmp_path / "no_viz_plan"
    circuit_input = CircuitInput(_config(), PROMPTS[:1])
    discoverer = CircuitDiscoverer(model, collector_class=CircuitCollector)

    plan = discoverer.plan(circuit_input, save_dir=str(save_dir), verbose=False)
    discoverer.apply(circuit_input, save_dir=str(save_dir), verbose=False)

    # The base collector writes the data and leaves out the page; visualization is opt-in via the subclass.
    assert "graph_0.graphml" in plan.outputs
    assert "circuit_0.html" not in plan.outputs
    assert sorted(path.name for path in save_dir.iterdir()) == sorted(plan.outputs)
