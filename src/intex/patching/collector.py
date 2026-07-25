import os
from collections.abc import Sequence

import pandas as pd
import torch

from intex.operation.collector import AbstractCollector
from intex.patching.input import PatchingInput
from intex.util.collections import dataframe_to_list


class PatchingCollector(AbstractCollector[PatchingInput]):
    def __init__(self, patching_input: PatchingInput, save_dir: str | None = None):
        super().__init__(patching_input, save_dir)

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
