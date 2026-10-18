from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from entoscope.model.config import ModelLocatorConfig
from entoscope.operation.config import AbstractConfig
from entoscope.operation.method import MethodConfig

if TYPE_CHECKING:
    from entoscope.patching.method.factory import PatchingMethodFactory


class PatchingType(StrEnum):
    """What a patch writes, named on its source element. A target element is always ``replace``.

    ``replace`` writes the source run's activation and ``value_scale`` the same times ``value``.
    ``value_replace`` writes the constant ``value``, zero ablation with the default 0.0, and
    ``gaussian_noise`` the target run's own activation plus noise at three times its spread, the
    corruption of causal tracing. Neither of those two reads the source run.
    """

    REPLACE = "replace"
    VALUE_REPLACE = "value_replace"
    VALUE_SCALE = "value_scale"
    GAUSSIAN_NOISE = "gaussian_noise"


class PatchingMetric(StrEnum):
    """The scalar read off one run's output distribution.

    ``r`` is the correct token and ``r'`` the distractor, both named by the input.
    ``kl_divergence`` needs neither: it scores the whole distribution against the source run's.
    """

    LOGIT_DIFF = "logit_diff"  # logit(r) - logit(r')
    PROBABILITY = "probability"  # P(r)
    LOG_PROB = "log_prob"  # log P(r)
    LOGPROB_DIFF = "logprob_diff"  # log P(r) - log P(r')
    KL_DIVERGENCE = "kl_divergence"  # D_KL(P_source || P_run)


class MetricComparison(StrEnum):
    """How a patched run's metric is reported against the two runs it sits between.

    Chosen per metric: a divergence is normally read as an ``indirect_effect`` and a logit
    difference as ``normalized``, and a run may name several for one metric.

    An ineffective patch reproduces the target run, and one that takes over entirely reproduces
    the source run, so ``normalized`` reads 0 and 1 at those two ends whichever of the pair is the
    clean prompt - that choice is expressed by the correct and distractor tokens, not here.
    """

    ABSOLUTE = "absolute"  # m(patched)
    INDIRECT_EFFECT = "indirect_effect"  # m(patched) - m(target)
    NORMALIZED = "normalized"  # (m(patched) - m(target)) / (m(source) - m(target))


@dataclass(kw_only=True)
class PatchingElementConfig(ModelLocatorConfig):
    """One side of a patch element: a locator, plus what the patch writes.

    Attributes:
        type: What the patch writes, on a source element, and ``replace`` on a target one.
        value: The constant ``value_replace`` writes and the factor ``value_scale`` multiplies by.
    """

    type: PatchingType
    value: float = 0.0


@dataclass
class SinglePatchingConfig:
    """A patch element: what is read at row ``i`` of the source index is written at row ``i`` of the target's.

    An index of a single row pairs with every row of the other. A config lists several elements to
    patch them together: patched run ``i`` applies row ``i`` of every element, so their indices have the
    same number of rows, or one. The result is laid out along the element with the most rows.

    Attributes:
        source: Where to read, and what to write.
        target: Where to write.
    """

    source: PatchingElementConfig
    target: PatchingElementConfig


MetricPair = tuple[PatchingMetric, MetricComparison]


def metric_column(metric: PatchingMetric, comparison: MetricComparison) -> str:
    """Name the result column of a metric reported under a comparison.

    Args:
        metric: The metric.
        comparison: What it is reported against.

    Returns:
        The metric's name for ``absolute``, otherwise ``<comparison>_<metric>``, such as
        ``normalized_logit_diff``.
    """
    if comparison is MetricComparison.ABSOLUTE:
        return metric.value
    return f"{comparison.value}_{metric.value}"


@dataclass(kw_only=True)
class PatchingMethodConfig(MethodConfig):
    """The method config of a patching method that takes no fields, such as ``exact``.

    A method with fields of its own names a dataclass extending this one.
    """

    @classmethod
    def factory(cls) -> type[PatchingMethodFactory]:
        """Name the factory patching methods register with.

        Returns:
            ``PatchingMethodFactory``.
        """
        from entoscope.patching.method.factory import PatchingMethodFactory

        return PatchingMethodFactory


@dataclass(kw_only=True)
class PatchingConfig(AbstractConfig):
    """What a patching run patches, and what it reads off every patched run.

    Every metric is read off the same patched run, so a second metric costs no extra forward pass and
    a second comparison of a metric already asked for costs nothing at all: it re-reads that metric's
    change. Each result column is drawn as its own plot on the result page.

    Attributes:
        metric: Each metric, mapped to the comparisons it is reported under, one result column each.
        method: How the change in a metric is obtained, its ``type`` naming the method. ``exact`` patches
            and runs the model once per index row, while ``attribution`` and ``integrated_gradients``
            estimate every row from gradients instead, trading a linearity assumption for a cost that
            does not grow with the index. In a YAML file it is the mapping read strictly into the method's
            config class, or just the method's type for its defaults.
        patch: The patch elements, applied together in every patched run.

    Raises:
        ValueError: If no metric is named, or a metric names no comparison or one twice.
    """

    metric: dict[PatchingMetric, list[MetricComparison]]
    method: PatchingMethodConfig
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

    def metric_pairs(self) -> list[MetricPair]:
        """Pair every metric with each comparison it is reported under.

        Returns:
            The pairs, in the order the config names them.
        """
        return [(metric, comparison) for metric, comparisons in self.metric.items() for comparison in comparisons]

    def metric_columns(self) -> list[str]:
        """Name the result columns.

        Returns:
            One column per metric pair, in the order the config names them.
        """
        return [metric_column(*pair) for pair in self.metric_pairs()]

    def primary(self) -> MetricPair:
        """Pick the run's headline metric and comparison.

        Returns:
            The first column's pair.
        """
        return self.metric_pairs()[0]
