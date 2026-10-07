import torch

from intex.lens.input import LensInput
from intex.lens.visualization import LensFigure
from intex.model.index import Index
from intex.model.token import str_token
from intex.operation.collector import FrameCollector, VisualizationCollector


class LensCollector(FrameCollector[LensInput]):
    def collect(self, logits: torch.Tensor, index: int):
        # Each cell holds the top-k tokens of a single-position slice; elements spanning
        # multiple positions (e.g. a layer-only location) do not fit this per-row layout.
        probabilities, tokens = logits.detach().softmax(dim=-1).squeeze(0).topk(self.operation_input.config.top_k)

        cell: dict[str, int | str | float] = {}
        for i, (token, probability) in enumerate(zip(tokens.tolist(), probabilities.tolist(), strict=True)):
            cell[f"token_{i}"] = token
            cell[f"str_token_{i}"] = str_token(self.model, token)
            cell[f"probability_{i}"] = probability
        self._collect_cell(index, cell)

    def _indices(self) -> dict[str, Index]:
        return {"index.csv": self.operation_input.index}


class LensVisualizationCollector(VisualizationCollector[LensInput, LensFigure], LensCollector):
    """A ``LensCollector`` that also draws its result as an interactive Plotly ``result.html``."""
