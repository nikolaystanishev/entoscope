from enum import Enum

from transformer_lens import HookedTransformer

from intex.model.config import ModelLocatorConfigMixin
from intex.operation.config import AbstractConfig
from intex.patching.index import PatchingIndex
from intex.patching.input import PatchingInput


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

    def prepare(self, model: HookedTransformer, patching_input: PatchingInput) -> PatchingIndex:
        return PatchingIndex(
            self.source.type,
            self.target.type,
            self.source.parse(model, patching_input.source_tokens.shape[1]),
            self.target.parse(model, patching_input.target_tokens.shape[1]),
        )


class PatchingConfig(AbstractConfig):
    def __init__(self, config_id: str, config: dict):
        self.config_id: str = config_id
        self.metric = PatchingMetric(config.get("metric"))
        self.patching_configs = list(map(SinglePatchingConfig, config.get("patch", [])))

    def prepare(self, model: HookedTransformer, patching_input: PatchingInput):
        patching_index = [patching_config.prepare(model, patching_input) for patching_config in self.patching_configs]

        index_lengths = [len(idx) for idx in patching_index]
        assert all(length == max(index_lengths) or length == 1 for length in index_lengths)

        patching_input.index = patching_index
