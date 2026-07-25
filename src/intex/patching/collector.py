import os
from collections.abc import Sequence

import torch

from intex.operation.collector import AbstractCollector
from intex.patching.input import PatchingInput

CHUNK_SIZE = 256


class DefaultCollector(AbstractCollector[PatchingInput]):
    def __init__(self, patching_input: PatchingInput, save_dir: str):
        super().__init__(patching_input, save_dir)

        self._result = None

        self.nan_tensor = torch.tensor([float("nan")])

    @property
    def result(self) -> torch.Tensor:
        return self._result

    def prepare(self):
        self._result = self.operation_input.index[0].prepare_output_placeholder()

    def collect(self, logits: torch.Tensor, index: Sequence[int] | None = None, metric_value: float = 0.0):
        self._result[index] = metric_value

    def _save_result(self, save_dir: str):
        torch.save(self._result, os.path.join(save_dir, "result.pt"))
        for i, idx in enumerate(self.operation_input.index):
            idx.to_csv(os.path.join(save_dir, f"index-{i}.csv"))

    def output_files(self) -> list[str]:
        indices = [f"index-{i}.csv" for i in range(len(self.operation_input.index))]
        return ["result.pt", *indices, *super().output_files()]
