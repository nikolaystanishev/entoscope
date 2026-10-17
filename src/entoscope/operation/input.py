from __future__ import annotations

import abc
from typing import Self

import torch

from entoscope.model import Model
from entoscope.operation.config import AbstractConfig


class OperationInput(abc.ABC):
    config: AbstractConfig

    @abc.abstractmethod
    def prepare_index(self, model: Model) -> Self:
        """Resolve everything that does not need a forward pass - what a run would iterate.

        Split out of ``prepare`` so an operation can be planned against a model without spending
        the compute.
        """
        raise NotImplementedError

    def prepare(self, model: Model) -> Self:
        """Resolve the index and gather the activations a run reads on top of it.

        An input that gathers none, its method running the model itself, is ready once its index is.
        """
        return self.prepare_index(model)

    @abc.abstractmethod
    def to(self, device: str | torch.device = "cpu") -> Self:
        raise NotImplementedError
