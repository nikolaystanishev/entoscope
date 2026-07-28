from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from intex.model.config import ModelLocatorConfig
from intex.operation.config import AbstractConfig


class PatchingType(str, Enum):
    REPLACE = "replace"
    VALUE_REPLACE = "value_replace"
    VALUE_SCALE = "value_scale"
    GAUSSIAN_NOISE = "gaussian_noise"


class PatchingMetric(str, Enum):
    """The scalar read off one run's output distribution.

    ``r`` is the correct token and ``r'`` the distractor, both named by the input.
    ``kl_divergence`` needs neither: it scores the whole distribution against the source run's.
    """

    LOGIT_DIFF = "logit_diff"  # logit(r) - logit(r')
    PROBABILITY = "probability"  # P(r)
    LOG_PROB = "log_prob"  # log P(r)
    LOGPROB_DIFF = "logprob_diff"  # log P(r) - log P(r')
    KL_DIVERGENCE = "kl_divergence"  # D_KL(P_source || P_run)


class MetricComparison(str, Enum):
    """How a patched run's metric is reported against the two runs it sits between.

    Chosen per metric: a divergence is normally read as an ``indirect_effect`` and a logit
    difference as ``normalized``, and a run may name several for one metric.

    An ineffective patch reproduces the target run, and one that takes over entirely reproduces
    the source run, so ``normalized`` reads 0 and 1 at those two ends whichever of the pair is the
    clean prompt — that choice is expressed by the correct and distractor tokens, not here.
    """

    ABSOLUTE = "absolute"  # m(patched)
    INDIRECT_EFFECT = "indirect_effect"  # m(patched) - m(target)
    NORMALIZED = "normalized"  # (m(patched) - m(target)) / (m(source) - m(target))


class PatchingMethodType(str, Enum):
    """How the change in the metric is obtained.

    ``exact`` patches and runs the model once per index row. The other two estimate every row from
    gradients instead, trading a linearity assumption for a cost that does not grow with the index.
    """

    EXACT = "exact"
    ATTRIBUTION = "attribution"
    INTEGRATED_GRADIENTS = "integrated_gradients"


@dataclass(kw_only=True)
class PatchingElementConfig(ModelLocatorConfig):
    type: PatchingType
    value: float = 0.0


@dataclass
class SinglePatchingConfig:
    source: PatchingElementConfig
    target: PatchingElementConfig


def metric_column(metric: PatchingMetric, comparison: MetricComparison) -> str:
    if comparison is MetricComparison.ABSOLUTE:
        return metric.value
    return f"{comparison.value}_{metric.value}"


@dataclass(kw_only=True)
class PatchingConfig(AbstractConfig):
    """``metric`` maps each metric to the comparisons it is reported under, one result column each.

    Everything named there is read off the same patched run, so a second metric costs no extra
    forward pass and a second comparison of a metric already asked for costs nothing at all — it
    re-reads that metric's change. The figure colours by the first column.

    ``interpolation_steps`` is the number of points the integrated-gradients method averages its
    gradient over and is ignored by the other two.
    """

    metric: dict[PatchingMetric, list[MetricComparison]]
    method: PatchingMethodType = PatchingMethodType.EXACT
    interpolation_steps: int = 10
    patch: list[SinglePatchingConfig] = field(default_factory=list)

    def __post_init__(self):
        if not self.metric:
            raise ValueError("At least one patching metric is required")
        for metric, comparisons in self.metric.items():
            if not comparisons:
                raise ValueError(f"{metric.value} names no comparison to report it under")
            repeated = sorted({c.value for c in comparisons if comparisons.count(c) > 1})
            if repeated:
                raise ValueError(f"{metric.value} names {', '.join(repeated)} more than once")
        if self.interpolation_steps < 1:
            raise ValueError(f"interpolation_steps must be at least 1, got {self.interpolation_steps}")

    def metric_columns(self) -> list[str]:
        """The result columns, in the order the config names them."""
        return [
            metric_column(metric, comparison)
            for metric, comparisons in self.metric.items()
            for comparison in comparisons
        ]

    def primary(self) -> tuple[PatchingMetric, MetricComparison]:
        """The first column's metric and comparison — what the figure is drawn from."""
        metric, comparisons = next(iter(self.metric.items()))
        return metric, comparisons[0]
