from __future__ import annotations

import abc
from typing import Self

import torch

from entoscope.model import Model
from entoscope.operation.config import AbstractConfig


class OperationInput(abc.ABC):
    """The base of every operation's input: a config together with the prompts it runs on.

    An input fills itself lazily through ``prepare``, which resolves what a run iterates and then
    gathers the activations it reads. Each of the two is done once, so ``plan`` followed by ``apply``
    on the same input resolves the index once.

    Attributes:
        config: The config the input runs.
    """

    config: AbstractConfig

    @abc.abstractmethod
    def prepare(self, model: Model, *, index_only: bool = False) -> Self:
        """Resolve the index a run iterates, then gather the activations it reads on top of it.

        The index needs no forward pass, so ``index_only`` lets an operation be planned against a model
        without spending the compute. An input that gathers no activations, its method running the
        model itself, is ready once its index is.

        Args:
            model: The model the run reads.
            index_only: Whether to stop after the index, before any forward pass.

        Returns:
            The input itself, prepared in place.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def to(self, device: str | torch.device = "cpu") -> Self:
        """Move the tensors the input holds to ``device``.

        Args:
            device: The device to move to.

        Returns:
            The input itself, moved in place.
        """
        raise NotImplementedError
