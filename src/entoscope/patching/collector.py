from entoscope.model.index import Index
from entoscope.model.token import str_token
from entoscope.operation.collector import FrameCollector, VisualizationCollector
from entoscope.patching.input import PatchingInput
from entoscope.patching.method.patching import RowResult
from entoscope.patching.visualization import PatchingFigure


class PatchingCollector(FrameCollector[PatchingInput]):
    """Writes the metrics of every patched run as ``result.csv``, plus one index file per element.

    A run that patched the model also records what it then predicts next, as ``generation`` (the
    token id, exact) and ``str_generation`` (its string, see ``str_token`` on reading it back).

    This base collector writes the CSV files only, and ``PatchingVisualizationCollector`` also draws
    ``result.html``.
    """

    def collect(self, row: RowResult):
        """Take one patched run's metrics.

        Args:
            row: The run's result row and metric columns, and its next token when the method ran the
                patched model.
        """
        cell: dict[str, float | int | str] = dict(row.metrics)
        if row.generation is not None:
            cell["generation"] = row.generation
            cell["str_generation"] = str_token(self.model, row.generation)
        self._collect_cell(row.output, cell)

    def _indices(self) -> dict[str, Index]:
        return {f"index-{i}.csv": index for i, index in enumerate(self.operation_input.index)}

    @property
    def _index(self) -> Index:
        return self.operation_input.layout_index


class PatchingVisualizationCollector(VisualizationCollector[PatchingInput, PatchingFigure], PatchingCollector):
    """A ``PatchingCollector`` that also draws its result as an interactive Plotly ``result.html``."""
