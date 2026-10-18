from __future__ import annotations

import abc
from typing import Self

import torch

from entoscope.model import Model
from entoscope.operation.config import AbstractConfig


class OperationInput(abc.ABC):
    """The base of every operation's input: a config together with the prompts it runs on.

    An input fills itself lazily. ``prepare_index`` resolves what a run iterates and ``prepare`` gathers
    the activations it reads, and both are idempotent, so ``plan`` followed by ``apply`` on the same
    input resolves the index once.

    Attributes:
        config: The config the input runs.
    """

    config: AbstractConfig

    @abc.abstractmethod
    def prepare_index(self, model: Model) -> Self:
        """Resolve everything that does not need a forward pass, which is what a run would iterate.

        Split out of ``prepare`` so an operation can be planned against a model without spending
        the compute.

        Args:
            model: The model the run reads.

        Returns:
            The input itself, prepared in place.
        """
        raise NotImplementedError

    def prepare(self, model: Model) -> Self:
        """Resolve the index and gather the activations a run reads on top of it.

        An input that gathers none, its method running the model itself, is ready once its index is.

        Args:
            model: The model the run reads.

        Returns:
            The input itself, prepared in place.
        """
        return self.prepare_index(model)

    @abc.abstractmethod
    def to(self, device: str | torch.device = "cpu") -> Self:
        """Move the tensors the input holds to ``device``.

        Args:
            device: The device to move to.

        Returns:
            The input itself, moved in place.
        """
        raise NotImplementedError
