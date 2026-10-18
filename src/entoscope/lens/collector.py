import torch

from entoscope.lens.input import LensInput
from entoscope.lens.visualization import LensFigure
from entoscope.model.index import Index
from entoscope.model.token import str_token
from entoscope.operation.collector import FrameCollector, VisualizationCollector


class LensCollector(FrameCollector[LensInput]):
    """One row per index element: its top-k tokens, best first.

    Rank ``i`` fills ``token_{i}`` (the token id), ``str_token_{i}`` (its string, escaped, see
    ``str_token`` on reading it back) and ``probability_{i}`` (the softmax over the element's logits).
    Writes ``result.csv``, ``index.csv`` and ``config.json``, and ``LensVisualizationCollector`` adds
    ``result.html``.
    """

    def collect(self, logits: torch.Tensor, index: int):
        """Rank one element's tokens.

        Args:
            logits: The element's logits over the vocabulary, ``[d_vocab]`` or ``[1, d_vocab]``.
            index: The element's row in the index.
        """
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
