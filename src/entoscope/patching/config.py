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

    ``r`` and ``r'`` are the first and second of the tokens a metric names in its ``tokens``.
    ``kl_divergence`` reads none: it scores the whole distribution against the source run's.
    """

    LOGIT_DIFF = "logit_diff"  # logit(r) - logit(r')
    PROBABILITY = "probability"  # P(r)
    LOG_PROB = "log_prob"  # log P(r)
    LOGPROB_DIFF = "logprob_diff"  # log P(r) - log P(r')
    KL_DIVERGENCE = "kl_divergence"  # D_KL(P_source || P_run)


class MetricComparison(StrEnum):
    """How a patched run's metric is reported against the two runs it sits between.

    Chosen per metric: a divergence is normally read as an ``indirect_effect`` and a logit
    difference as ``normalized``.

    An ineffective patch reproduces the target run, and one that takes over entirely reproduces
    the source run, so ``normalized`` reads 0 and 1 at those two ends whichever of the pair is the
    clean prompt - that choice is expressed by the tokens a metric reads and their order, not here.
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


@dataclass(kw_only=True)
class PatchingMetricConfig:
    """A metric a run reads off every patched run, the tokens it reads and what it is reported against.

    The config's ``metric`` names it, so several metrics of one type can each read tokens of their own,
    or be reported under a comparison of their own.

    Attributes:
        type: The scalar read off a run.
        tokens: The tokens the metric reads, by their names in the input's ``tokens``: two for a
            difference, the first read against the second, one for ``probability`` and ``log_prob``, and
            none for ``kl_divergence``.
        comparison: What the metric is reported against.
    """

    type: PatchingMetric
    tokens: list[str] = field(default_factory=list)
    comparison: MetricComparison


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

    Every metric is read off the same patched run, so a second metric costs no extra forward pass. Each
    metric's result column is drawn as its own plot on the result page.

    Attributes:
        metric: Each metric, by a name of the config's choosing, which names its result column too.
        method: How the change in a metric is obtained, its ``type`` naming the method. ``exact`` patches
            and runs the model once per index row, while ``attribution`` and ``integrated_gradients``
            estimate every row from gradients instead, trading a linearity assumption for a cost that
            does not grow with the index. In a YAML file it is the mapping read strictly into the method's
            config class, or just the method's type for its defaults.
        patch: The patch elements, applied together in every patched run.
    """

    metric: dict[str, PatchingMetricConfig]
    method: PatchingMethodConfig
    patch: list[SinglePatchingConfig] = field(default_factory=list)
