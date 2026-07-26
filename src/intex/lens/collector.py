import os
from collections.abc import Sequence

import pandas as pd
import torch

from intex.lens.input import LensInput
from intex.lens.visualization import LensFigure
from intex.model import Model
from intex.operation.collector import AbstractCollector
from intex.util.collections import dataframe_to_list


def escape_token(token: str) -> str:
    """Escape non-printable characters (e.g. "\\r", "\\n" byte tokens) — raw ones break CSV round-trips."""
    return "".join(char if char.isprintable() else char.encode("unicode_escape").decode("ascii") for char in token)


class LensCollector(AbstractCollector[LensInput]):
    def __init__(self, lens_input: LensInput, model: Model, save_dir: str | None = None):
        super().__init__(lens_input, save_dir)
        self.model = model

        self._cells: dict[tuple[int, ...], dict[str, int | str | float]] | None = None

    @property
    def result(self) -> pd.DataFrame:
        assert self._cells is not None
        rows = pd.DataFrame(
            [self._cells[tuple(row)] for row in dataframe_to_list(self.operation_input.index.output_df)]
        )
        return pd.concat([self.operation_input.index.to_df().reset_index(drop=True), rows], axis=1)

    def prepare(self):
        self._cells = {}

    def collect(self, logits: torch.Tensor, index: Sequence[int]):
        assert self._cells is not None
        # Each cell holds the top-k tokens of a single-position slice; elements spanning
        # multiple positions (e.g. a layer-only location) do not fit this per-row layout.
        probabilities, tokens = logits.detach().softmax(dim=-1).squeeze(0).topk(self.operation_input.config.top_k)

        cell: dict[str, int | str | float] = {}
        for i, (token, probability) in enumerate(zip(tokens.tolist(), probabilities.tolist(), strict=True)):
            cell[f"token_{i}"] = token
            cell[f"str_token_{i}"] = escape_token(self.model.to_single_str_token(token))
            cell[f"probability_{i}"] = probability
        self._cells[tuple(index)] = cell

    def _save_result(self, save_dir: str):
        self.result.to_csv(os.path.join(save_dir, "result.csv"), index=False)
        self.operation_input.index.to_csv(os.path.join(save_dir, "index.csv"))

    def output_files(self) -> list[str]:
        return ["result.csv", "index.csv", *super().output_files()]


class LensVisualizationCollector(LensCollector):
    """A ``LensCollector`` that also writes an interactive Plotly ``result.html``.

    Kept separate from the base collector so the base (and anything that only needs the CSV) does
    not pull in plotly, and so visualization is opt-in by collector choice rather than a config flag.
    """

    def _save_result(self, save_dir: str):
        super()._save_result(save_dir)
        self._render_html(self.result, self.operation_input, self._str_tokens(), os.path.join(save_dir, "result.html"))

    @staticmethod
    def _render_html(result: pd.DataFrame, lens_input: LensInput, tokens: dict[str, list[str]], path: str) -> None:
        LensFigure(result, lens_input, tokens).render(path)

    def output_files(self) -> list[str]:
        return ["result.html", *super().output_files()]

    def _str_tokens(self) -> dict[str, list[str]]:
        # The parameter lens reads the weights, so it has no prompt and no tokens to label an axis with.
        if self.operation_input.prompt is None:
            return {}
        tokens = self.model.to_str_tokens(self.operation_input.tokens[0])
        return {"source_position": [escape_token(token) for token in tokens]}
