import os

import networkx as nx

from intex.circuit.input import CircuitInput
from intex.circuit.method.circuit import PromptCircuit
from intex.circuit.visualization import CircuitFigure
from intex.operation.collector import AbstractCollector, FigureMixin
from intex.operation.config import CONFIG_FILE
from intex.util.data import write_json


def _write_graph(graph: nx.MultiDiGraph, save_dir: str, name: str) -> None:
    """Persist a graph as both GraphML (portable) and node-link JSON (readable / round-trippable)."""
    nx.write_graphml(graph, os.path.join(save_dir, f"{name}.graphml"))
    write_json(os.path.join(save_dir, f"{name}.json"), nx.node_link_data(graph, edges="links"))


class CircuitCollector(AbstractCollector[CircuitInput]):
    """Persists the discovered graphs (GraphML + node-link JSON) and a ``prompts.json`` index.

    This base collector writes graph data only. Use ``CircuitVisualizationCollector`` to also render
    an interactive HTML per prompt.
    """

    _circuits: list[PromptCircuit] | None = None

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
        self.operation_input.config.save(save_dir)

    def output_files(self) -> list[str]:
        graphs = [
            f"{name}_{i}.{extension}"
            for i in range(len(self.operation_input.prompts))
            for name in ("graph", "routes")
            for extension in ("graphml", "json")
        ]
        return [*graphs, "prompts.json", CONFIG_FILE, *super().output_files()]


class CircuitVisualizationCollector(CircuitCollector, FigureMixin[CircuitFigure]):
    """A ``CircuitCollector`` that also draws each prompt's circuit as ``circuit_{i}.html``.

    Kept separate from the base collector so the base (and anything that only needs graph data) does
    not pull in plotly, and so visualization is opt-in by collector choice rather than a config flag.
    """

    def _save_result(self, save_dir: str) -> None:
        super()._save_result(save_dir)
        for i, circuit in enumerate(self.result):
            self.figure_class(circuit, self.model).render(os.path.join(save_dir, f"circuit_{i}.html"))

    def output_files(self) -> list[str]:
        pages = [f"circuit_{i}.html" for i in range(len(self.operation_input.prompts))]
        return [*pages, *super().output_files()]
