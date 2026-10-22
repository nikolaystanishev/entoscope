from __future__ import annotations

import abc
import logging
from typing import TYPE_CHECKING, ClassVar

import torch
from jaxtyping import Float

from entoscope.patching.config import MetricComparison, PatchingMetric

if TYPE_CHECKING:
    from entoscope.patching.input import PatchingInput

Logits = Float[torch.Tensor, "d_vocab"]

logger = logging.getLogger(__name__)


class AbstractMetric(abc.ABC):
    """One scalar read off a run's final-position logits, on the tokens it reads.

    It holds no run of its own: the input, which holds the two unpatched runs, is handed to it with every
    run it reads.

    Args:
        tokens: The ids of the tokens the metric reads, in the order its config names them.

    Attributes:
        effect_sign: The sign an ``indirect_effect`` is reported with, -1 for a metric that falls as the
            patch restores the source run.
        n_tokens: How many tokens the metric reads, which its config names in ``tokens``.
        needs_source_run: Whether the metric scores against the source run.
        tokens: The ids of the tokens the metric reads.
    """

    effect_sign: ClassVar[int] = 1
    n_tokens: ClassVar[int] = 0
    needs_source_run: ClassVar[bool] = False

    def __init__(self, tokens: list[int]):
        self.tokens = tokens

    def source(self, patching_input: PatchingInput) -> torch.Tensor:
        """Read the metric off the unpatched source run.

        Args:
            patching_input: The prepared input, which has a source prompt.

        Returns:
            The metric, a scalar.
        """
        return self.compute(patching_input, patching_input.source_logits[0, -1])

    def target(self, patching_input: PatchingInput) -> torch.Tensor:
        """Read the metric off the unpatched target run.

        Args:
            patching_input: The prepared input.

        Returns:
            The metric, a scalar.
        """
        return self.compute(patching_input, patching_input.target_logits[0, -1])

    @abc.abstractmethod
    def compute(self, patching_input: PatchingInput, logits: Logits) -> torch.Tensor:
        """Read the metric off one run, differentiably, so the gradient methods can back-propagate through it.

        Args:
            patching_input: The prepared input, whose unpatched runs a metric may score against.
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

    n_tokens = 2

    def compute(self, patching_input: PatchingInput, logits: Logits) -> torch.Tensor:
        token, against = self.tokens
        return logits[token] - logits[against]


class ProbabilityMetric(AbstractMetric):
    """``P(r)``."""

    n_tokens = 1

    def compute(self, patching_input: PatchingInput, logits: Logits) -> torch.Tensor:
        (token,) = self.tokens
        return logits.softmax(dim=-1)[token]


class LogProbMetric(AbstractMetric):
    """``log P(r)``."""

    n_tokens = 1

    def compute(self, patching_input: PatchingInput, logits: Logits) -> torch.Tensor:
        (token,) = self.tokens
        return logits.log_softmax(dim=-1)[token]


class LogProbDiffMetric(AbstractMetric):
    """``log P(r) - log P(r')``.

    Numerically the same thing as ``LogitDiffMetric``, since the softmax's normalizer is shared by the
    two tokens and cancels, and kept apart because the papers that use it call it by this name.
    """

    n_tokens = 2

    def compute(self, patching_input: PatchingInput, logits: Logits) -> torch.Tensor:
        token, against = self.tokens
        log_probabilities = logits.log_softmax(dim=-1)
        return log_probabilities[token] - log_probabilities[against]


class KLDivergenceMetric(AbstractMetric):
    """``D_KL(P_source || P_run)``, how far a run's whole distribution sits from the source run's.

    Needs no answer token, which is what makes it the task-agnostic choice. ``kl_div(input, target)``
    reads ``input`` as log-probabilities and computes ``D_KL(target || exp(input))``, so the run
    being scored is the one passed as ``input``.
    """

    effect_sign = -1
    needs_source_run = True

    def compute(self, patching_input: PatchingInput, logits: Logits) -> torch.Tensor:
        return torch.nn.functional.kl_div(
            logits.log_softmax(dim=-1), patching_input.source_logits[0, -1].softmax(dim=-1), reduction="sum"
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
    def apply(cls, metric: AbstractMetric, patching_input: PatchingInput, delta: torch.Tensor) -> torch.Tensor:
        """Report a change in a metric.

        Args:
            metric: The metric that changed.
            patching_input: The prepared input, whose unpatched runs the change is reported against.
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
    def apply(cls, metric: AbstractMetric, patching_input: PatchingInput, delta: torch.Tensor) -> torch.Tensor:
        return metric.target(patching_input) + delta


class IndirectEffectComparison(AbstractComparison):
    """The patched run's metric minus the unpatched target run's, signed by the metric's ``effect_sign``."""

    @classmethod
    def apply(cls, metric: AbstractMetric, patching_input: PatchingInput, delta: torch.Tensor) -> torch.Tensor:
        return metric.effect_sign * delta


class NormalizedComparison(AbstractComparison):
    """0 where the patch changed nothing, 1 where it reproduced the source run.

    The same ratio circuit work calls faithfulness, with the unpatched target run playing the part
    of the fully-ablated model. Two runs that score the same leave no gap to divide by: every row then
    reads inf, or nan where the patch changed nothing, and a warning is logged.
    """

    needs_source_run = True

    @classmethod
    def apply(cls, metric: AbstractMetric, patching_input: PatchingInput, delta: torch.Tensor) -> torch.Tensor:
        target = metric.target(patching_input)
        gap = metric.source(patching_input) - target
        if gap == 0:
            logger.warning(
                "A normalized metric divides by the gap between the source and target runs, which both score %.6g: "
                "the row reads inf, or nan where the patch changed nothing",
                float(target),
            )
        return delta / gap
