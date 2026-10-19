from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from entoscope.model.config import ModelAxis, ModelLocatorConfig
from entoscope.model.slicer import located_axes
from entoscope.operation.config import AbstractConfig
from entoscope.operation.method import MethodConfig

if TYPE_CHECKING:
    from entoscope.probe.method.factory import ProbeMethodFactory


class PoolingType(StrEnum):
    """How ``ProbeConfig.pool`` reduces an axis. ``last`` keeps its final entry, the last token for a position axis."""

    MEAN = "mean"
    MAX = "max"
    SUM = "sum"
    LAST = "last"


class ProbeMetric(StrEnum):
    """A score of a probe's predictions on its examples.

    ``auroc`` ranks the positive-class probabilities, and the others compare the 0/1 predictions with
    the labels.
    """

    ACCURACY = "accuracy"
    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"
    AUROC = "auroc"


@dataclass(kw_only=True)
class ProbeMethodConfig(MethodConfig):
    """The method config of a probe that takes no fields.

    A probe with fields of its own names a dataclass extending this one.
    """

    @classmethod
    def factory(cls) -> type[ProbeMethodFactory]:
        """Name the factory probe methods register with.

        Returns:
            ``ProbeMethodFactory``.
        """
        from entoscope.probe.method.factory import ProbeMethodFactory

        return ProbeMethodFactory


@dataclass(kw_only=True)
class ProbeConfig(AbstractConfig):
    """What a probing run trains, where, and how it is scored.

    Attributes:
        method: The probe trained at every location, its ``type`` naming it. In a YAML file it is the
            mapping read strictly into the probe's method config class, or just the probe's type for its
            defaults.
        pool: Each axis to reduce, mapped to the pooling that reduces it, matched by name: ``attn.pattern``'s
            positions are pooled as ``dest_pos`` and ``src_pos``. Axes left out are flattened into the
            feature vector. ``layer`` is never a tensor axis and ``neuron`` is the feature axis, so neither
            can be pooled.
        metric: The scores to report for every probe.
        source: Where to read the features.

    Raises:
        ValueError: If ``pool`` names ``layer`` or ``neuron``, or an axis no activation of ``source`` has.
    """

    method: ProbeMethodConfig
    pool: dict[ModelAxis, PoolingType] = field(default_factory=dict)
    metric: list[ProbeMetric]
    source: ModelLocatorConfig

    def __post_init__(self):
        super().__post_init__()
        unpoolable = {ModelAxis.LAYER, ModelAxis.NEURON} & set(self.pool)
        if unpoolable:
            raise ValueError(f"Locations {sorted(location.value for location in unpoolable)} cannot be pooled")

        # An entry for an axis no row's tensor has would reduce nothing: the axis meant, such as the pattern's
        # positions for 'position', would stay in the features without a word.
        activations = self.source.activations()
        if not activations:
            return
        poolable = set().union(*map(located_axes, activations)) - {ModelAxis.LAYER, ModelAxis.NEURON}
        for axis in self.pool:
            if axis not in poolable:
                names = ", ".join(sorted(f"'{activation.value}'" for activation in activations))
                axes = ", ".join(f"'{name.value}'" for name in ModelAxis if name in poolable)
                raise ValueError(f"'{axis.value}' is not an axis of {names} to pool: the poolable axes are {axes}")
