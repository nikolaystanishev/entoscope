from intex.model.index import Index
from intex.operation.collector import FrameCollector, VisualizationCollector
from intex.patching.input import PatchingInput
from intex.patching.visualization import PatchingFigure


class PatchingCollector(FrameCollector[PatchingInput]):
    """Persists the metrics of every patch run as ``result.csv``, plus one index file per element.

    This base collector writes the CSVs only. Use ``PatchingVisualizationCollector`` to also render
    an interactive HTML.
    """

    def collect(self, index: int, metric_values: dict[str, float]):
        self._collect_cell(index, dict(metric_values))

    def _indices(self) -> dict[str, Index]:
        # Numbered even for a single element, so a file name means the same thing in every run.
        return {f"index-{i}.csv": index for i, index in enumerate(self.operation_input.index)}


class PatchingVisualizationCollector(VisualizationCollector[PatchingInput, PatchingFigure], PatchingCollector):
    """A ``PatchingCollector`` that also draws its result as an interactive Plotly ``result.html``."""
