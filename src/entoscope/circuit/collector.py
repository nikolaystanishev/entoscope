import os

from entoscope.circuit.input import CircuitInput
from entoscope.circuit.method.circuit import PromptCircuit
from entoscope.circuit.visualization import CircuitFigure
from entoscope.model import Model
from entoscope.operation.collector import AbstractCollector, FigureMixin
from entoscope.operation.config import CONFIG_FILE
from entoscope.util.data import write_graph, write_json


class CircuitCollector(AbstractCollector[CircuitInput]):
    """Persists the discovered graphs (GraphML + node-link JSON) and a ``prompts.json`` index.

    This base collector writes graph data only. Use ``CircuitVisualizationCollector`` to also render
    an interactive HTML per prompt.
    """

    def __init__(self, operation_input: CircuitInput, model: Model, save_dir: str | None = None):
        super().__init__(operation_input, model, save_dir)
        self._circuits: list[PromptCircuit] = []

    @property
    def result(self) -> list[PromptCircuit]:
        return self._circuits

    def collect(self, circuit: PromptCircuit, index: int) -> None:
        assert index == len(self._circuits)
        self._circuits.append(circuit)

    def _save_result(self, save_dir: str) -> None:
        summary = []
        for i, circuit in enumerate(self.result):
            write_graph(circuit.graph, save_dir, f"graph_{i}")
            write_graph(circuit.routes, save_dir, f"routes_{i}")
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

    Kept separate from the base collector so that anything which only needs graph data draws no
    figure, and so visualization is opt-in by collector choice rather than a config flag.
    """

    def _save_result(self, save_dir: str) -> None:
        super()._save_result(save_dir)
        for i, circuit in enumerate(self.result):
            self.figure_class(circuit, self.model, self.operation_input.config).render(
                os.path.join(save_dir, f"circuit_{i}.html")
            )

    def output_files(self) -> list[str]:
        pages = [f"circuit_{i}.html" for i in range(len(self.operation_input.prompts))]
        return [*pages, *super().output_files()]
