from __future__ import annotations

import abc
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from entoscope.model.index import IndexElement, SingleIndex
from entoscope.patching.config import PatchingElementConfig, PatchingType

if TYPE_CHECKING:
    from entoscope.patching.input import PatchingInput

_CACHE_BATCH = 0


class SourceStrategy(abc.ABC):
    """What a patch writes, by the ``type`` of its source element.

    Attributes:
        needs_source_run: Whether the strategy reads the source run.
    """

    needs_source_run: bool = False

    @staticmethod
    @abc.abstractmethod
    def extract(index: IndexElement[PatchingElementConfig], patching_input: PatchingInput) -> torch.Tensor:
        """Produce the value a patch writes.

        Args:
            index: The source element.
            patching_input: The prepared input.

        Returns:
            The value, shaped as the source element's slice or broadcastable to the target's.
        """
        raise NotImplementedError

    @staticmethod
    def get(patching_type: PatchingType) -> type[SourceStrategy]:
        """Pick the strategy of a source element's type.

        Args:
            patching_type: The source element's type.

        Returns:
            The strategy.

        Raises:
            ValueError: If the type is unknown.
        """
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
    """How a patch is written, by the ``type`` of its target element."""

    @staticmethod
    @abc.abstractmethod
    def set(
        index: IndexElement[PatchingElementConfig],
        target_activation: torch.Tensor,
        extracted_source: torch.Tensor,
        batch: int,
    ) -> torch.Tensor:
        """Write a value into the patched run's activation.

        Args:
            index: The target element.
            target_activation: The patched run's activation, written in place.
            extracted_source: What the source strategy produced.
            batch: The batch row of the patched run.

        Returns:
            The patched activation.
        """
        raise NotImplementedError

    @staticmethod
    def get(patching_type: PatchingType) -> type[TargetStrategy]:
        """Pick the strategy of a target element's type.

        Args:
            patching_type: The target element's type.

        Returns:
            The strategy.

        Raises:
            ValueError: If the type is not ``replace``.
        """
        match patching_type:
            case PatchingType.REPLACE:
                return ReplaceTargetStrategy
            case _:
                raise ValueError(f"Unknown target strategy: {patching_type}")


class ReplaceSourceStrategy(SourceStrategy):
    """The source run's activation."""

    needs_source_run = True

    @staticmethod
    def extract(source_index: IndexElement[PatchingElementConfig], patching_input: PatchingInput) -> torch.Tensor:
        return source_index.read_slices(
            patching_input.source_cache[source_index.transformer_activation()], _CACHE_BATCH
        )


class ValueReplaceSourceStrategy(SourceStrategy):
    """The element's constant ``value``."""

    @staticmethod
    def extract(source_index: IndexElement[PatchingElementConfig], patching_input: PatchingInput) -> torch.Tensor:
        return torch.tensor(source_index.config.value)


class ValueScaleSourceStrategy(SourceStrategy):
    """The source run's activation times the element's ``value``."""

    needs_source_run = True

    @staticmethod
    def extract(source_index: IndexElement[PatchingElementConfig], patching_input: PatchingInput) -> torch.Tensor:
        return (
            source_index.read_slices(patching_input.source_cache[source_index.transformer_activation()], _CACHE_BATCH)
            * source_index.config.value
        )


class GaussianNoiseSourceStrategy(SourceStrategy):
    """The target's own activation plus ``N(0, noise_scale)`` noise, the corruption of causal tracing.

    Seeded the same on every call, so each row of a sweep is corrupted by the same draw and differs
    from the others only in where it lands, and a rerun repeats the sweep exactly.

    Attributes:
        SEED: The seed every draw starts from.
    """

    SEED = 42

    @staticmethod
    def extract(source_index: IndexElement[PatchingElementConfig], patching_input: PatchingInput) -> torch.Tensor:
        name = source_index.transformer_activation()
        clean = source_index.read_slices(patching_input.target_cache[name], _CACHE_BATCH)
        generator = torch.Generator().manual_seed(GaussianNoiseSourceStrategy.SEED)
        noise = torch.randn(clean.shape, generator=generator).to(clean)

        return clean + noise * patching_input.noise_scale(name)


class ReplaceTargetStrategy(TargetStrategy):
    """Overwrites the target element's slice with the value."""

    @staticmethod
    def set(
        index: IndexElement[PatchingElementConfig],
        target_activation: torch.Tensor,
        extracted_source: torch.Tensor,
        batch: int,
    ) -> torch.Tensor:
        index.write_slices(target_activation, extracted_source, batch)
        return target_activation


class PatchProcessor:
    """Applies one patch, source strategy then target strategy."""

    @staticmethod
    def patch(
        index: SingleIndex[PatchingElementConfig],
        patching_input: PatchingInput,
        target_activation: torch.Tensor,
        batch: int,
    ) -> torch.Tensor:
        """Write a row's source into one batch row of the patched run's activation.

        Args:
            index: The row to patch.
            patching_input: The prepared input.
            target_activation: The patched run's activation, written in place.
            batch: The batch row of the patched run.

        Returns:
            The patched activation.
        """
        return TargetStrategy.get(index.target.config.type).set(
            index.target,
            target_activation,
            SourceStrategy.get(index.source.config.type).extract(index.source, patching_input),
            batch,
        )


class HookedPatchProcessor(PatchProcessor):
    """Applies a patch as a ``transformer_lens`` forward hook."""

    @classmethod
    def hook(
        cls, index: SingleIndex[PatchingElementConfig], patching_input: PatchingInput, batch: int
    ) -> tuple[str, Callable]:
        """Build the forward hook that patches one row.

        Args:
            index: The row to patch.
            patching_input: The prepared input.
            batch: The batch row of the patched run.

        Returns:
            The hook name and the hook function, as ``run_with_hooks`` takes them.
        """
        return index.target.transformer_activation(), lambda target_activation, hook: cls.patch(
            index, patching_input, target_activation, batch
        )
