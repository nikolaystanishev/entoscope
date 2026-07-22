from __future__ import annotations

import abc

import torch

from intex.operation.config import AbstractConfig


class OperationInput(abc.ABC):
    config: AbstractConfig

    @abc.abstractmethod
    def to(self, device: str | torch.device = "cpu") -> OperationInput:
        raise NotImplementedError
