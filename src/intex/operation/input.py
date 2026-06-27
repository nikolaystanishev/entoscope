from __future__ import annotations

import abc

import torch


class OperationInput(abc.ABC):
    @abc.abstractmethod
    def to(self, device: str | torch.device = "cpu") -> OperationInput:
        raise NotImplementedError
