from intex.model.index import Index
from intex.model.token import str_token
from intex.operation.collector import FrameCollector, VisualizationCollector
from intex.patching.input import PatchingInput
from intex.patching.method.patching import RowResult
from intex.patching.visualization import PatchingFigure


class PatchingCollector(FrameCollector[PatchingInput]):
    """Persists the metrics of every patch run as ``result.csv``, plus one index file per element.

    A run that patched the model also records what it then predicts next, as ``generation`` (the
    token id, exact) and ``str_generation`` (its string, see ``str_token`` on reading it back).

    This base collector writes the CSVs only. Use ``PatchingVisualizationCollector`` to also render
    an interactive HTML.
    """

    def collect(self, row: RowResult):
        cell: dict[str, float | int | str] = dict(row.metrics)
        if row.generation is not None:
            cell["generation"] = row.generation
            cell["str_generation"] = str_token(self.model, row.generation)
        self._collect_cell(row.output, cell)

    def _indices(self) -> dict[str, Index]:
        # Numbered even for a single element, so a file name means the same thing in every run.
        return {f"index-{i}.csv": index for i, index in enumerate(self.operation_input.index)}


class PatchingVisualizationCollector(VisualizationCollector[PatchingInput, PatchingFigure], PatchingCollector):
    """A ``PatchingCollector`` that also draws its result as an interactive Plotly ``result.html``."""
