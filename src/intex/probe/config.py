from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from intex.model.config import ModelLocation, ModelLocatorConfig
from intex.operation.config import AbstractConfig
from intex.operation.method import MethodConfig

if TYPE_CHECKING:
    from intex.probe.method.factory import ProbeMethodFactory


class PoolingType(str, Enum):
    """How ``ProbeConfig.pool`` reduces an axis; ``last`` keeps its final entry - the last token, for ``position``."""

    MEAN = "mean"
    MAX = "max"
    SUM = "sum"
    LAST = "last"


class ProbeMetric(str, Enum):
    """A score of a probe's predictions on its examples: ``auroc`` ranks the positive-class probabilities,
    the others compare the 0/1 predictions with the labels."""

    ACCURACY = "accuracy"
    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"
    AUROC = "auroc"


@dataclass(kw_only=True)
class ProbeMethodConfig(MethodConfig):
    """The method config of a probe that takes no fields; a probe with fields names a dataclass extending this
    one."""

    @classmethod
    def factory(cls) -> type[ProbeMethodFactory]:
        # Imported here: the factory imports the probes, which import this module.
        from intex.probe.method.factory import ProbeMethodFactory

        return ProbeMethodFactory


@dataclass(kw_only=True)
class ProbeConfig(AbstractConfig):
    """``method`` is the probe trained at every location, ``type`` naming it: an instance of its method config class,
    or in a YAML file the mapping read strictly into that class, or just the probe's type for its defaults.
    ``pool`` maps a location axis to the pooling that
    reduces it; unlisted axes are flattened into the feature vector. ``layer`` is never a tensor axis and
    ``neuron`` is the feature axis, so neither can be pooled."""

    method: ProbeMethodConfig
    pool: dict[ModelLocation, PoolingType] = field(default_factory=dict)
    metric: list[ProbeMetric]
    source: ModelLocatorConfig

    def __post_init__(self):
        unpoolable = {ModelLocation.LAYER, ModelLocation.NEURON} & set(self.pool)
        if unpoolable:
            raise ValueError(f"Locations {sorted(location.value for location in unpoolable)} cannot be pooled")
