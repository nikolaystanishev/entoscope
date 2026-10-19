from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import torch
from jaxtyping import Float

from entoscope.patching.config import MetricComparison, PatchingMetric

if TYPE_CHECKING:
    from entoscope.patching.input import PatchingInput

Logits = Float[torch.Tensor, "d_vocab"]


class AbstractMetric(abc.ABC):
    """One scalar read off a single run's final-position logits.

    Attributes:
        effect_sign: The sign an ``indirect_effect`` is reported with, -1 for a metric that falls as the
            patch restores the source run.
        needs_distractor: Whether the metric reads the distractor token.
        needs_source_run: Whether the metric scores against the source run.
    """

    effect_sign: int = 1
    needs_distractor: bool = False
    needs_source_run: bool = False

    @classmethod
    @abc.abstractmethod
    def compute(cls, patching_input: PatchingInput, logits: Logits) -> torch.Tensor:
        """Read the metric off one run, differentiably, so the gradient methods can back-propagate through it.

        Args:
            patching_input: The prepared input, which names the answer tokens.
            logits: The run's logits at the final position.

        Returns:
            The metric, a scalar.
        """
        raise NotImplementedError

    @staticmethod
    def get(metric: PatchingMetric) -> type[AbstractMetric]:
        """Pick the implementation of a metric.

        Args:
            metric: The metric.

        Returns:
            The metric's class.

        Raises:
            ValueError: If the metric is unknown.
        """
        match metric:
            case PatchingMetric.LOGIT_DIFF:
                return LogitDiffMetric
            case PatchingMetric.PROBABILITY:
                return ProbabilityMetric
            case PatchingMetric.LOG_PROB:
                return LogProbMetric
            case PatchingMetric.LOGPROB_DIFF:
                return LogProbDiffMetric
            case PatchingMetric.KL_DIVERGENCE:
                return KLDivergenceMetric
            case _:
                raise ValueError(f"Unknown patching metric: {metric}")


class LogitDiffMetric(AbstractMetric):
    """``logit(r) - logit(r')``."""

    needs_distractor = True

    @classmethod
    def compute(cls, patching_input: PatchingInput, logits: Logits) -> torch.Tensor:
        return logits[patching_input.correct_token_id] - logits[patching_input.distractor_token_id]


class ProbabilityMetric(AbstractMetric):
    """``P(r)``."""

    @classmethod
    def compute(cls, patching_input: PatchingInput, logits: Logits) -> torch.Tensor:
        return logits.softmax(dim=-1)[patching_input.correct_token_id]


class LogProbMetric(AbstractMetric):
    """``log P(r)``."""

    @classmethod
    def compute(cls, patching_input: PatchingInput, logits: Logits) -> torch.Tensor:
        return logits.log_softmax(dim=-1)[patching_input.correct_token_id]


class LogProbDiffMetric(AbstractMetric):
    """``log P(r) - log P(r')``.

    Numerically the same thing as ``LogitDiffMetric``, since the softmax's normalizer is shared by the
    two tokens and cancels, and kept apart because the papers that use it call it by this name.
    """

    needs_distractor = True

    @classmethod
    def compute(cls, patching_input: PatchingInput, logits: Logits) -> torch.Tensor:
        log_probabilities = logits.log_softmax(dim=-1)
        return (
            log_probabilities[patching_input.correct_token_id] - log_probabilities[patching_input.distractor_token_id]
        )


class KLDivergenceMetric(AbstractMetric):
    """``D_KL(P_source || P_run)``, how far a run's whole distribution sits from the source run's.

    Needs no answer token, which is what makes it the task-agnostic choice. ``kl_div(input, target)``
    reads ``input`` as log-probabilities and computes ``D_KL(target || exp(input))``, so the run
    being scored is the one passed as ``input``.
    """

    effect_sign = -1
    needs_source_run = True

    @classmethod
    def compute(cls, patching_input: PatchingInput, logits: Logits) -> torch.Tensor:
        return torch.nn.functional.kl_div(
            logits.log_softmax(dim=-1), patching_input.source_probabilities, reduction="sum"
        )


class AbstractComparison(abc.ABC):
    """Turns the change a patch made to a metric into the number a run reports.

    Every method produces the same primitive, ``delta``, the patched run's metric minus the unpatched
    target run's: exactly for ``exact`` and as a gradient estimate for the others.

    Attributes:
        needs_source_run: Whether the comparison scores against the source run.
    """

    needs_source_run: bool = False

    @classmethod
    @abc.abstractmethod
    def apply(cls, metric: PatchingMetric, patching_input: PatchingInput, delta: torch.Tensor) -> torch.Tensor:
        """Report a change in a metric.

        Args:
            metric: The metric that changed.
            patching_input: The prepared input, which holds the unpatched runs' metrics.
            delta: The patched run's metric minus the unpatched target run's.

        Returns:
            The reported number.
        """
        raise NotImplementedError

    @staticmethod
    def get(comparison: MetricComparison) -> type[AbstractComparison]:
        """Pick the implementation of a comparison.

        Args:
            comparison: The comparison.

        Returns:
            The comparison's class.

        Raises:
            ValueError: If the comparison is unknown.
        """
        match comparison:
            case MetricComparison.ABSOLUTE:
                return AbsoluteComparison
            case MetricComparison.INDIRECT_EFFECT:
                return IndirectEffectComparison
            case MetricComparison.NORMALIZED:
                return NormalizedComparison
            case _:
                raise ValueError(f"Unknown metric comparison: {comparison}")


class AbsoluteComparison(AbstractComparison):
    """The patched run's metric itself."""

    @classmethod
    def apply(cls, metric: PatchingMetric, patching_input: PatchingInput, delta: torch.Tensor) -> torch.Tensor:
        return patching_input.target_baseline(metric) + delta


class IndirectEffectComparison(AbstractComparison):
    """The patched run's metric minus the unpatched target run's, signed by the metric's ``effect_sign``."""

    @classmethod
    def apply(cls, metric: PatchingMetric, patching_input: PatchingInput, delta: torch.Tensor) -> torch.Tensor:
        return AbstractMetric.get(metric).effect_sign * delta


class NormalizedComparison(AbstractComparison):
    """0 where the patch changed nothing, 1 where it reproduced the source run.

    The same ratio circuit work calls faithfulness, with the unpatched target run playing the part
    of the fully-ablated model.
    """

    needs_source_run = True

    @classmethod
    def apply(cls, metric: PatchingMetric, patching_input: PatchingInput, delta: torch.Tensor) -> torch.Tensor:
        return delta / (patching_input.source_baseline(metric) - patching_input.target_baseline(metric))
