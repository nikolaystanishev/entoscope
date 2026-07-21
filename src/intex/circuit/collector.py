import os

import networkx as nx
from transformer_lens import HookedTransformer

from intex.circuit.input import CircuitInput
from intex.circuit.method.circuit import PromptCircuit
from intex.circuit.visualization import render_circuit_html
from intex.operation.collector import AbstractCollector
from intex.util.data import write_json


def _write_graph(graph: nx.MultiDiGraph, save_dir: str, name: str) -> None:
    """Persist a graph as both GraphML (portable) and node-link JSON (readable / round-trippable)."""
    nx.write_graphml(graph, os.path.join(save_dir, f"{name}.graphml"))
    write_json(os.path.join(save_dir, f"{name}.json"), nx.node_link_data(graph, edges="links"))


class CircuitCollector(AbstractCollector[CircuitInput]):
    """Persists the discovered graphs (GraphML + node-link JSON) and a ``prompts.json`` index.

    This base collector writes graph data only. Use ``CircuitVisualizationCollector`` to also render
    an interactive HTML per prompt. ``model`` is unused here but kept in the signature so the operator
    can construct either collector uniformly (the subclass needs it for ``cfg.n_layers``).
    """

    def __init__(self, circuit_input: CircuitInput, model: HookedTransformer, save_dir: str | None = None):
        super().__init__(circuit_input, save_dir)
        self.model = model

        self._circuits: list[PromptCircuit] | None = None

    @property
    def result(self) -> list[PromptCircuit]:
        assert self._circuits is not None
        return self._circuits

    def prepare(self) -> None:
        self._circuits = []

    def collect(self, circuit: PromptCircuit, index: int) -> None:
        assert self._circuits is not None
        assert index == len(self._circuits)
        self._circuits.append(circuit)

    def _save_result(self, save_dir: str) -> None:
        summary = []
        for i, circuit in enumerate(self.result):
            _write_graph(circuit.graph, save_dir, f"graph_{i}")
            _write_graph(circuit.routes, save_dir, f"routes_{i}")
            summary.append(
                {
                    "index": i,
                    "prompt": circuit.prompt,
                    "str_tokens": circuit.str_tokens,
                    "graph_edges": circuit.graph.number_of_edges(),
                    "route_edges": circuit.routes.number_of_edges(),
                }
            )
        write_json(os.path.join(save_dir, "prompts.json"), summary)


class CircuitVisualizationCollector(CircuitCollector):
    """A ``CircuitCollector`` that also writes an interactive Plotly ``circuit_{i}.html`` per prompt.

    Kept separate from the base collector so the base (and anything that only needs graph data) does
    not pull in plotly, and so visualization is opt-in by collector choice rather than a config flag.
    """

    def _save_result(self, save_dir: str) -> None:
        super()._save_result(save_dir)
        for i, circuit in enumerate(self.result):
            render_circuit_html(circuit, self.model.cfg.n_layers, os.path.join(save_dir, f"circuit_{i}.html"))
