from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from intex.model.config import ModelLocation, ModelLocatorConfig
from intex.operation.config import AbstractConfig
from intex.util.data import from_dict, to_dict


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
class ProbeSettings:
    """The settings of a probe that takes none; a probe with settings names a dataclass extending this one."""


@dataclass(kw_only=True)
class ProbeConfig(AbstractConfig):
    """``method`` names the probe trained at every location, and ``settings`` are its own, given as an instance
    of its ``settings_class`` or as the mapping a YAML file gives, and read strictly into that class as the
    config is built. ``pool`` maps a location axis to the pooling that reduces it; unlisted axes are flattened
    into the feature vector. ``layer`` is never a tensor axis and ``neuron`` is the feature axis, so neither
    can be pooled."""

    method: str
    pool: dict[ModelLocation, PoolingType] = field(default_factory=dict)
    metric: list[ProbeMetric]
    source: ModelLocatorConfig
    settings: ProbeSettings | dict[str, Any] = field(default_factory=ProbeSettings)

    def __post_init__(self):
        unpoolable = {ModelLocation.LAYER, ModelLocation.NEURON} & set(self.pool)
        if unpoolable:
            raise ValueError(f"Locations {sorted(location.value for location in unpoolable)} cannot be pooled")
        # Imported here: the factory imports the probes, which import this module.
        from intex.probe.method.factory import ProbeMethodFactory

        settings_class = ProbeMethodFactory.get(self.method).settings_class
        # Read through its raw shape, so the empty default takes the probe's defaults and another probe's
        # settings - which also extend ProbeSettings - cannot pass as this probe's.
        self.settings = from_dict(settings_class, to_dict(self.settings))
