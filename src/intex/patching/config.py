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


@dataclass(kw_only=True)
class PatchingConfig(AbstractConfig):
    """``metric`` is a list because every one of them is read off the same patched run: asking for
    several costs no extra forward pass, and the figure colours by the first.

    ``interpolation_steps`` is the number of points the integrated-gradients method averages its
    gradient over and is ignored by the other two.
    """

    metric: list[PatchingMetric]
    comparison: MetricComparison = MetricComparison.ABSOLUTE
    method: PatchingMethodType = PatchingMethodType.EXACT
    interpolation_steps: int = 10
    patch: list[SinglePatchingConfig] = field(default_factory=list)

    def __post_init__(self):
        if not self.metric:
            raise ValueError("At least one patching metric is required")
        if self.interpolation_steps < 1:
            raise ValueError(f"interpolation_steps must be at least 1, got {self.interpolation_steps}")

    def metric_columns(self) -> dict[PatchingMetric, str]:
        """The result column each metric fills, in the order the config names them."""
        return {metric: self.metric_column(metric) for metric in self.metric}

    def metric_column(self, metric: PatchingMetric) -> str:
        if self.comparison is MetricComparison.ABSOLUTE:
            return metric.value
        return f"{self.comparison.value}_{metric.value}"
