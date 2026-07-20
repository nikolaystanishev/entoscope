from dataclasses import dataclass, field
from enum import Enum

from intex.model.config import ModelLocation, ModelLocatorConfig
from intex.operation.config import AbstractConfig


class ProbeType(str, Enum):
    LINEAR = "linear"


class PoolingType(str, Enum):
    MEAN = "mean"
    MAX = "max"
    SUM = "sum"
    LAST = "last"


class ProbeMetric(str, Enum):
    ACCURACY = "accuracy"
    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"
    AUROC = "auroc"


@dataclass(kw_only=True)
class ProbeConfig(AbstractConfig):
    """``pool`` maps a location axis to the pooling that reduces it; unlisted axes are
    flattened into the feature vector. ``layer`` is never a tensor axis and ``neuron``
    is the feature axis, so neither can be pooled."""

    type: ProbeType
    pool: dict[ModelLocation, PoolingType] = field(default_factory=dict)
    metric: list[ProbeMetric]
    source: ModelLocatorConfig
    max_iter: int = 1000
    seed: int = 42

    def __post_init__(self):
        unpoolable = {ModelLocation.LAYER, ModelLocation.NEURON} & set(self.pool)
        if unpoolable:
            raise ValueError(f"Locations {sorted(location.value for location in unpoolable)} cannot be pooled")
