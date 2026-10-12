from __future__ import annotations

import abc

import torch

from intex.model import Model
from intex.operation.config import AbstractConfig


class OperationInput(abc.ABC):
    config: AbstractConfig

    @abc.abstractmethod
    def prepare_index(self, model: Model) -> OperationInput:
        """Resolve everything that does not need a forward pass - what a run would iterate.

        Split out of ``prepare`` so an operation can be planned against a model without spending
        the compute; ``prepare`` calls it and then gathers the activations on top.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def prepare(self, model: Model) -> OperationInput:
        raise NotImplementedError

    @abc.abstractmethod
    def to(self, device: str | torch.device = "cpu") -> OperationInput:
        raise NotImplementedError
