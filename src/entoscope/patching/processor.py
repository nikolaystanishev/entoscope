import abc
from collections.abc import Callable

import torch

from entoscope.model.index import IndexElement, SingleIndex
from entoscope.patching.config import PatchingElementConfig, PatchingType
from entoscope.patching.input import PatchingInput

# A patched run holds one copy of the target prompt per row, so a row's batch index addresses that
# run alone. The caches are made as wide as the patched runs but kept at their first row, so every
# cache holds a single prompt and is read at batch 0.
_CACHE_BATCH = 0


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
        index: IndexElement[PatchingElementConfig],
        target_activation: torch.Tensor,
        extracted_source: torch.Tensor,
        batch: int,
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
        return patching_input.source_cache[source_index.transformer_activation()][source_index.slices(_CACHE_BATCH)]


class ValueReplaceSourceStrategy(SourceStrategy):
    @staticmethod
    def extract(source_index: IndexElement[PatchingElementConfig], patching_input: PatchingInput) -> torch.Tensor:
        return torch.tensor(source_index.config.value)


class ValueScaleSourceStrategy(SourceStrategy):
    @staticmethod
    def extract(source_index: IndexElement[PatchingElementConfig], patching_input: PatchingInput) -> torch.Tensor:
        return (
            patching_input.source_cache[source_index.transformer_activation()][source_index.slices(_CACHE_BATCH)]
            * source_index.config.value
        )


class GaussianNoiseSourceStrategy(SourceStrategy):
    """The target's own activation plus ``N(0, noise_scale)`` noise - the corruption of causal tracing.

    Seeded the same on every call, so each row of a sweep is corrupted by the same draw and differs
    from the others only in where it lands, and a rerun repeats the sweep exactly.
    """

    SEED = 42

    @staticmethod
    def extract(source_index: IndexElement[PatchingElementConfig], patching_input: PatchingInput) -> torch.Tensor:
        name = source_index.transformer_activation()
        clean = patching_input.target_cache[name][source_index.slices(_CACHE_BATCH)]
        # Drawn on the CPU, so a seed gives the same noise whatever device the model runs on.
        generator = torch.Generator().manual_seed(GaussianNoiseSourceStrategy.SEED)
        noise = torch.randn(clean.shape, generator=generator).to(clean)

        return clean + noise * patching_input.noise_scale(name)


class ReplaceTargetStrategy(TargetStrategy):
    @staticmethod
    def set(
        index: IndexElement[PatchingElementConfig],
        target_activation: torch.Tensor,
        extracted_source: torch.Tensor,
        batch: int,
    ) -> torch.Tensor:
        target_activation[index.slices(batch)] = extracted_source
        return target_activation


class PatchProcessor:
    @staticmethod
    def patch(
        index: SingleIndex[PatchingElementConfig],
        patching_input: PatchingInput,
        target_activation: torch.Tensor,
        batch: int,
    ) -> torch.Tensor:
        """Write ``index``'s source into row ``batch`` of the patched run's ``target_activation``."""
        return TargetStrategy.get(index.target.config.type).set(
            index.target,
            target_activation,
            SourceStrategy.get(index.source.config.type).extract(index.source, patching_input),
            batch,
        )


class HookedPatchProcessor(PatchProcessor):
    @classmethod
    def hook(
        cls, index: SingleIndex[PatchingElementConfig], patching_input: PatchingInput, batch: int
    ) -> tuple[str, Callable]:
        return index.target.transformer_activation(), lambda target_activation, hook: cls.patch(
            index, patching_input, target_activation, batch
        )
