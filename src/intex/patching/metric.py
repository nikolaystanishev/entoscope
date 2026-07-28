from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import torch
from jaxtyping import Float

from intex.patching.config import MetricComparison, PatchingMetric

if TYPE_CHECKING:
    from intex.patching.input import PatchingInput

Logits = Float[torch.Tensor, "d_vocab"]


class AbstractMetric(abc.ABC):
    """One scalar read off a single run's final-position logits.

    ``compute`` stays differentiable so the gradient methods can back-propagate through it.
    """

    # The metric's effect is normally the rise in m, but a KL divergence is a distance, so its
    # effect is the drop — that keeps 'positive means the patch restored the source behaviour'
    # true for every metric. It cancels in the normalized comparison, which divides by the same sign.
    effect_sign: int = 1
    needs_distractor: bool = False
    needs_source_run: bool = False

    @classmethod
    @abc.abstractmethod
    def compute(cls, patching_input: PatchingInput, logits: Logits) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def get(metric: PatchingMetric) -> type[AbstractMetric]:
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
    needs_distractor = True

    @classmethod
    def compute(cls, patching_input: PatchingInput, logits: Logits) -> torch.Tensor:
        return logits[patching_input.correct_token_id] - logits[patching_input.distractor_token_id]


class ProbabilityMetric(AbstractMetric):
    @classmethod
    def compute(cls, patching_input: PatchingInput, logits: Logits) -> torch.Tensor:
        return logits.softmax(dim=-1)[patching_input.correct_token_id]


class LogProbMetric(AbstractMetric):
    @classmethod
    def compute(cls, patching_input: PatchingInput, logits: Logits) -> torch.Tensor:
        return logits.log_softmax(dim=-1)[patching_input.correct_token_id]


class LogProbDiffMetric(AbstractMetric):
    """Numerically the same thing as ``LogitDiffMetric`` — the softmax's normalizer is shared by the
    two tokens and cancels — and kept apart because the papers that use it call it by this name."""

    needs_distractor = True

    @classmethod
    def compute(cls, patching_input: PatchingInput, logits: Logits) -> torch.Tensor:
        log_probabilities = logits.log_softmax(dim=-1)
        return (
            log_probabilities[patching_input.correct_token_id] - log_probabilities[patching_input.distractor_token_id]
        )


class KLDivergenceMetric(AbstractMetric):
    """``D_KL(P_source || P_run)`` — how far a run's whole distribution sits from the source run's.

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

    Every method produces the same primitive — ``delta``, the patched run's metric minus the
    unpatched target run's — exactly for ``exact`` and as a gradient estimate for the others.
    """

    needs_source_run: bool = False

    @classmethod
    @abc.abstractmethod
    def apply(cls, metric: PatchingMetric, patching_input: PatchingInput, delta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def get(comparison: MetricComparison) -> type[AbstractComparison]:
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
    @classmethod
    def apply(cls, metric: PatchingMetric, patching_input: PatchingInput, delta: torch.Tensor) -> torch.Tensor:
        return patching_input.target_baseline(metric) + delta


class IndirectEffectComparison(AbstractComparison):
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
