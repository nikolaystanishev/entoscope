from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from intex.model.config import ModelLocatorConfig
from intex.operation.config import AbstractConfig
from intex.util.data import from_dict, to_dict


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


@dataclass(kw_only=True)
class PatchingElementConfig(ModelLocatorConfig):
    type: PatchingType
    value: float = 0.0


@dataclass
class SinglePatchingConfig:
    source: PatchingElementConfig
    target: PatchingElementConfig


MetricPair = tuple[PatchingMetric, MetricComparison]


def metric_column(metric: PatchingMetric, comparison: MetricComparison) -> str:
    if comparison is MetricComparison.ABSOLUTE:
        return metric.value
    return f"{comparison.value}_{metric.value}"


@dataclass(kw_only=True)
class PatchingSettings:
    """The settings of a method that takes none; a method with settings names a dataclass extending this one."""


@dataclass(kw_only=True)
class PatchingConfig(AbstractConfig):
    """``metric`` maps each metric to the comparisons it is reported under, one result column each.

    Everything named there is read off the same patched run, so a second metric costs no extra
    forward pass and a second comparison of a metric already asked for costs nothing at all — it
    re-reads that metric's change. Each column is drawn as its own plot on the result page.

    ``method`` names how that change is obtained: ``exact`` patches and runs the model once per index
    row, while ``attribution`` and ``integrated_gradients`` estimate every row from gradients instead,
    trading a linearity assumption for a cost that does not grow with the index. ``settings`` are the
    method's own, given as an instance of its ``settings_class`` or as the mapping a YAML file gives, and
    read strictly into that class as the config is built.
    """

    metric: dict[PatchingMetric, list[MetricComparison]]
    method: str = "exact"
    patch: list[SinglePatchingConfig] = field(default_factory=list)
    settings: PatchingSettings | dict[str, Any] = field(default_factory=PatchingSettings)

    def __post_init__(self):
        if not self.metric:
            raise ValueError("At least one patching metric is required")
        for metric, comparisons in self.metric.items():
            if not comparisons:
                raise ValueError(f"{metric.value} names no comparison to report it under")
            repeated = sorted({c.value for c in comparisons if comparisons.count(c) > 1})
            if repeated:
                raise ValueError(f"{metric.value} names {', '.join(repeated)} more than once")
        # Imported here: the factory imports the methods, which import this module.
        from intex.patching.method.factory import PatchingMethodFactory

        settings_class = PatchingMethodFactory.get(self.method).settings_class
        # Read through its raw shape, so the empty default takes the method's defaults and another method's
        # settings - which also extend PatchingSettings - cannot pass as this method's.
        self.settings = from_dict(settings_class, to_dict(self.settings))

    def metric_pairs(self) -> list[MetricPair]:
        """Every metric with each comparison it is reported under, in the order the config names them."""
        return [(metric, comparison) for metric, comparisons in self.metric.items() for comparison in comparisons]

    def metric_columns(self) -> list[str]:
        """The result columns, in the order the config names them."""
        return [metric_column(*pair) for pair in self.metric_pairs()]

    def primary(self) -> MetricPair:
        """The first column's metric and comparison — the run's headline pair."""
        return self.metric_pairs()[0]
