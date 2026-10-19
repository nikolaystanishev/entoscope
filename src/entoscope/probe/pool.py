import abc
from collections.abc import Sequence

import torch
from jaxtyping import Float

from entoscope.model.config import ModelAxis
from entoscope.probe.config import PoolingType


class AbstractPooling(abc.ABC):
    """Reduces one axis of an activation."""

    @classmethod
    @abc.abstractmethod
    def reduce(cls, activation: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
        """Reduce one axis.

        Args:
            activation: The activation.
            dim: The axis to reduce.

        Returns:
            The activation without that axis.
        """
        raise NotImplementedError

    @staticmethod
    def get(pooling_type: PoolingType) -> type["AbstractPooling"]:
        """Pick the implementation of a pooling type.

        Args:
            pooling_type: The pooling type.

        Returns:
            The pooling's class.

        Raises:
            ValueError: If the pooling type is unknown.
        """
        match pooling_type:
            case PoolingType.MEAN:
                return MeanPooling
            case PoolingType.MAX:
                return MaxPooling
            case PoolingType.SUM:
                return SumPooling
            case PoolingType.LAST:
                return LastPooling
            case _:
                raise ValueError(f"Unknown pooling type: {pooling_type}")


class MeanPooling(AbstractPooling):
    """The mean along the axis."""

    @classmethod
    def reduce(cls, activation: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
        return activation.mean(dim=dim)


class MaxPooling(AbstractPooling):
    """The maximum along the axis."""

    @classmethod
    def reduce(cls, activation: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
        return activation.amax(dim=dim)


class SumPooling(AbstractPooling):
    """The sum along the axis."""

    @classmethod
    def reduce(cls, activation: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
        return activation.sum(dim=dim)


class LastPooling(AbstractPooling):
    """The axis's final entry."""

    @classmethod
    def reduce(cls, activation: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
        return activation.select(dim, -1)


def pool(
    activation: Float[torch.Tensor, "..."],
    axes: Sequence[ModelAxis | str],
    config: dict[ModelAxis, PoolingType],
) -> Float[torch.Tensor, " features"]:
    """Reduce each axis the config names with its pooling type, and flatten the rest into features.

    Configured axes are reduced left to right, which matters when different pooling types are mixed.
    Config entries that match no axis are ignored, such as an axis the location grid already pins.

    Args:
        activation: One example's activation at one location.
        axes: The activation's axes, in order.
        config: Each axis to reduce, mapped to its pooling type.

    Returns:
        The feature vector, a single feature for a 0-D activation.
    """
    if activation.ndim == 0:
        return activation.reshape(1)
    assert len(axes) == activation.ndim

    dim = 0
    for axis in axes:
        if axis in config:
            activation = AbstractPooling.get(config[axis]).reduce(activation, dim)
        else:
            dim += 1

    return activation.flatten()
