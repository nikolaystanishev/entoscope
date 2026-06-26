from __future__ import annotations

from enum import Enum

from intex.model.config import ModelLocatorConfigMixin
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


class PatchingElementConfig(ModelLocatorConfigMixin):
    def __init__(self, config: dict):
        ModelLocatorConfigMixin.__init__(self, config)
        self.type = PatchingType(config["type"])

        self.value: float = config.get("value", 0.0)


class SinglePatchingConfig:
    def __init__(self, config: dict):
        self.source = PatchingElementConfig(config["source"])
        self.target = PatchingElementConfig(config["target"])


class PatchingConfig(AbstractConfig):
    def __init__(self, config_id: str, config: dict):
        self.config_id: str = config_id
        self.metric = PatchingMetric(config.get("metric"))
        self.patching_configs = list(map(SinglePatchingConfig, config.get("patch", [])))
