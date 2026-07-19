import abc
from collections.abc import Sequence

import torch
from jaxtyping import Float

from intex.model.config import ModelLocation
from intex.probe.config import PoolingType


class AbstractPooling(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def reduce(cls, activation: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
        raise NotImplementedError

    @staticmethod
    def get(pooling_type: PoolingType) -> type["AbstractPooling"]:
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
    @classmethod
    def reduce(cls, activation: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
        return activation.mean(dim=dim)


class MaxPooling(AbstractPooling):
    @classmethod
    def reduce(cls, activation: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
        return activation.amax(dim=dim)


class SumPooling(AbstractPooling):
    @classmethod
    def reduce(cls, activation: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
        return activation.sum(dim=dim)


class LastPooling(AbstractPooling):
    @classmethod
    def reduce(cls, activation: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
        return activation.select(dim, -1)


def _matches(axis: ModelLocation | str, location: ModelLocation) -> bool:
    # The pattern activation splits position into dest/src axes; both belong to POSITION.
    return axis == location or (location is ModelLocation.POSITION and axis in {"dest_pos", "src_pos"})


def pool(
    activation: Float[torch.Tensor, "..."],
    axes: Sequence[ModelLocation | str],
    config: dict[ModelLocation, PoolingType],
) -> Float[torch.Tensor, " features"]:
    """Reduce each axis named in ``config`` with its pooling type and flatten the rest.

    ``axes`` names ``activation``'s axes in order. Configured axes are reduced left to
    right — the order matters when different pooling types are mixed. Config entries that
    match no axis are ignored (e.g. an axis already pinned by the location grid). 0-D
    activations become a single feature.
    """
    if activation.ndim == 0:
        return activation.reshape(1)
    assert len(axes) == activation.ndim

    dim = 0
    for axis in axes:
        location = next((location for location in config if _matches(axis, location)), None)
        if location is None:
            dim += 1  # axis kept: the next axis sits one dim further right
        else:
            activation = AbstractPooling.get(config[location]).reduce(activation, dim)

    return activation.flatten()
