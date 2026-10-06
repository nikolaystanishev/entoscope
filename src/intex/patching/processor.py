import abc
from collections.abc import Callable

import numpy as np
import torch

from intex.model.index import IndexElement, SingleIndex
from intex.patching.config import PatchingElementConfig, PatchingType
from intex.patching.input import PatchingInput


def _cached_slices(index: IndexElement[PatchingElementConfig]) -> tuple[slice | int | list[int], ...]:
    """``index``'s slices into a cached run, which holds one prompt.

    Only the patched run is batched - one copy of the target prompt per row - so the batch index an
    element carries addresses that run alone; every cache is read at batch 0.
    """
    return (0, *index.slices()[1:])


class SourceStrategy(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def extract(index: IndexElement[PatchingElementConfig], patching_input: PatchingInput) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def get(patching_type: PatchingType) -> type["SourceStrategy"]:
        match patching_type:
            case PatchingType.REPLACE:
                return ReplaceSourceStrategy
            case PatchingType.VALUE_REPLACE:
                return ValueReplaceSourceStrategy
            case PatchingType.VALUE_SCALE:
                return ValueScaleSourceStrategy
            case PatchingType.GAUSSIAN_NOISE:
                return GaussianNoiseSourceStrategy
            case _:
                raise ValueError(f"Unknown source strategy: {patching_type}")


class TargetStrategy(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def set(
        index: IndexElement[PatchingElementConfig], target_activation: torch.Tensor, extracted_source: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def get(patching_type: PatchingType) -> type["TargetStrategy"]:
        match patching_type:
            case PatchingType.REPLACE:
                return ReplaceTargetStrategy
            case _:
                raise ValueError(f"Unknown target strategy: {patching_type}")


class ReplaceSourceStrategy(SourceStrategy):
    @staticmethod
    def extract(source_index: IndexElement[PatchingElementConfig], patching_input: PatchingInput) -> torch.Tensor:
        return patching_input.source_cache[source_index.transformer_activation()][_cached_slices(source_index)]


class ValueReplaceSourceStrategy(SourceStrategy):
    @staticmethod
    def extract(source_index: IndexElement[PatchingElementConfig], patching_input: PatchingInput) -> torch.Tensor:
        return torch.tensor(source_index.config.value)


class ValueScaleSourceStrategy(SourceStrategy):
    @staticmethod
    def extract(source_index: IndexElement[PatchingElementConfig], patching_input: PatchingInput) -> torch.Tensor:
        return (
            patching_input.source_cache[source_index.transformer_activation()][_cached_slices(source_index)]
            * source_index.config.value
        )


class GaussianNoiseSourceStrategy(SourceStrategy):
    @staticmethod
    def extract(source_index: IndexElement[PatchingElementConfig], patching_input: PatchingInput) -> torch.Tensor:
        noise_level = 3 * patching_input.target_cache[source_index.transformer_activation()].std()
        target_activations = patching_input.target_cache[source_index.transformer_activation()][
            _cached_slices(source_index)
        ]

        return (
            target_activations
            * torch.from_numpy(np.random.RandomState(42).rand(*target_activations.shape))
            .to(target_activations.dtype)
            .to(target_activations.device)
            * noise_level
        )


class ReplaceTargetStrategy(TargetStrategy):
    @staticmethod
    def set(
        index: IndexElement[PatchingElementConfig], target_activation: torch.Tensor, extracted_source: torch.Tensor
    ) -> torch.Tensor:
        target_activation[index.slices()] = extracted_source
        return target_activation


class PatchProcessor:
    @staticmethod
    def patch(
        index: SingleIndex[PatchingElementConfig],
        patching_input: PatchingInput,
        target_activation: torch.Tensor,
    ) -> torch.Tensor:
        return TargetStrategy.get(index.target.config.type).set(
            index.target,
            target_activation,
            SourceStrategy.get(index.source.config.type).extract(index.source, patching_input),
        )


class HookedPatchProcessor(PatchProcessor):
    @classmethod
    def hook(cls, index: SingleIndex[PatchingElementConfig], patching_input: PatchingInput) -> tuple[str, Callable]:
        return index.target.transformer_activation(), lambda target_activation, hook: cls.patch(
            index, patching_input, target_activation
        )
