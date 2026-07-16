import abc
import os
from collections.abc import Sequence

import torch

from intex.patching.input import PatchingInput
from intex.util.collections import to_dict
from intex.util.data import create_if_not_exists, write_json

CHUNK_SIZE = 256


class AbstractCollector(abc.ABC):
    def __init__(self, patching_input: PatchingInput, save_dir: str | None = None):
        self.patching_input = patching_input
        self.save_dir = save_dir

    @property
    @abc.abstractmethod
    def result(self) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def prepare(self):
        raise NotImplementedError

    @abc.abstractmethod
    def collect(self, logits: torch.Tensor, index: Sequence[int] | None = None, metric_value: float = 0.0):
        raise NotImplementedError

    @abc.abstractmethod
    def save(self):
        raise NotImplementedError


class DefaultCollector(AbstractCollector):
    def __init__(self, patching_input: PatchingInput, save_dir: str):
        super().__init__(patching_input, save_dir)

        self._result = None

        self.nan_tensor = torch.tensor([float("nan")])

    @property
    def result(self) -> torch.Tensor:
        return self._result

    def prepare(self):
        self._result = self.patching_input.index[0].prepare_output_placeholder()

    def collect(self, logits: torch.Tensor, index: Sequence[int] | None = None, metric_value: float = 0.0):
        self._result[index] = metric_value

    def save(self):
        if self.save_dir is not None:
            create_if_not_exists(self.save_dir)

            torch.save(self._result, os.path.join(self.save_dir, "result.pt"))
            for i, idx in enumerate(self.patching_input.index):
                idx.to_csv(os.path.join(self.save_dir, f"index-{i}.csv"))
            write_json(os.path.join(self.save_dir, "config.json"), to_dict(self.patching_input.config))

            print(f"{self.patching_input.config.config_id} done - results saved to {self.save_dir}")
