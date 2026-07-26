import os
from collections.abc import Sequence

import pandas as pd
import torch

from intex.lens.collector import escape_token
from intex.model import Model
from intex.operation.collector import AbstractCollector
from intex.patching.input import PatchingInput
from intex.patching.visualization import PatchingFigure
from intex.util.collections import dataframe_to_list


class PatchingCollector(AbstractCollector[PatchingInput]):
    """Persists the metric of every patch run as ``result.csv``, plus one index file per element.

    This base collector writes the CSVs only. Use ``PatchingVisualizationCollector`` to also render
    an interactive HTML. ``model`` is unused here but kept in the signature so the operator can
    construct either collector uniformly (the subclass needs it to name the prompt's tokens).
    """

    def __init__(self, patching_input: PatchingInput, model: Model, save_dir: str | None = None):
        super().__init__(patching_input, save_dir)
        self.model = model

        self._cells: dict[tuple[int, ...], dict[str, float]] | None = None

    @property
    def result(self) -> pd.DataFrame:
        assert self._cells is not None
        # The first patch element lays out the result; the rest only say what else each run patched.
        index = self.operation_input.index[0]
        rows = pd.DataFrame([self._cells[tuple(row)] for row in dataframe_to_list(index.output_df)])
        return pd.concat([index.to_df().reset_index(drop=True), rows], axis=1)

    def prepare(self):
        self._cells = {}

    def collect(self, logits: torch.Tensor, index: Sequence[int], metric_value: float):
        assert self._cells is not None
        self._cells[tuple(index)] = {self.operation_input.config.metric.value: metric_value}

    def _save_result(self, save_dir: str):
        self.result.to_csv(os.path.join(save_dir, "result.csv"), index=False)
        for i, idx in enumerate(self.operation_input.index):
            idx.to_csv(os.path.join(save_dir, f"index-{i}.csv"))

    def output_files(self) -> list[str]:
        indices = [f"index-{i}.csv" for i in range(len(self.operation_input.index))]
        return ["result.csv", *indices, *super().output_files()]


class PatchingVisualizationCollector(PatchingCollector):
    """A ``PatchingCollector`` that also writes an interactive Plotly ``result.html``.

    Kept separate from the base collector so the base (and anything that only needs the CSV) does
    not pull in plotly, and so visualization is opt-in by collector choice rather than a config flag.
    """

    def _save_result(self, save_dir: str):
        super()._save_result(save_dir)
        self._render_html(self.result, self.operation_input, self._str_tokens(), os.path.join(save_dir, "result.html"))

    @staticmethod
    def _render_html(
        result: pd.DataFrame, patching_input: PatchingInput, tokens: dict[str, list[str]], path: str
    ) -> None:
        PatchingFigure(result, patching_input, tokens).render(path)

    def output_files(self) -> list[str]:
        return ["result.html", *super().output_files()]

    def _str_tokens(self) -> dict[str, list[str]]:
        return {
            f"{prefix}position": [escape_token(token) for token in self.model.to_str_tokens(tokens[0])]
            for prefix, tokens in (
                ("target_", self.operation_input.target_tokens),
                ("source_", self.operation_input.source_tokens),
            )
        }
