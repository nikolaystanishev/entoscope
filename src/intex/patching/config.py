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
    LOGIT_DIFF = "logit_diff"
    NORMALIZED_LOGIT_DIFF = "normalized_logit_diff"
    KL_DIFF = "kl_diff"


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
    metric: PatchingMetric
    patch: list[SinglePatchingConfig] = field(default_factory=list)
