from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy

from transformer_lens.utilities import get_act_name

from intex.model.config import ModelLayerLocation, ModelLocation, ModelLocatorConfig
from intex.model.slicer import ActivationSlicer

# Maps intex's dotted ``ModelLayerLocation`` values onto the names ``get_act_name`` expects.
TRANSFORMER_ACT_NAME: dict[ModelLayerLocation, str] = {
    ModelLayerLocation.EMBEDDING: "embed",
    ModelLayerLocation.RESIDUAL_PRE: "resid_pre",
    ModelLayerLocation.RESIDUAL_MID: "resid_mid",
    ModelLayerLocation.RESIDUAL_POST: "resid_post",
    ModelLayerLocation.ATTENTION_Q: "q",
    ModelLayerLocation.ATTENTION_K: "k",
    ModelLayerLocation.ATTENTION_V: "v",
    ModelLayerLocation.ATTENTION_PATTERN: "pattern",
    ModelLayerLocation.ATTENTION_Z: "z",
    ModelLayerLocation.ATTENTION_OUT: "attn_out",
    ModelLayerLocation.MLP_PRE: "pre",
    ModelLayerLocation.MLP_PRE_LINEAR: "pre_linear",
    ModelLayerLocation.MLP_POST: "post",
    ModelLayerLocation.MLP_OUT: "mlp_out",
}


class IndexElement:
    def __init__(self, config: ModelLocatorConfig, index_names: list[str], index: list[int | list[int] | str]):
        self.config = config
        self.index_names = deepcopy(index_names)
        self.index = index
        self.batch_index = None

    def slices(self) -> tuple[slice | int | list[int], ...]:
        index: defaultdict[str, int | list[int] | slice] = defaultdict(lambda: slice(None))
        index.update(
            {name: idx for name, idx in zip(self.index_names, self.index, strict=True) if isinstance(idx, str) is False}
        )

        if self.batch_index is not None:
            index["batch"] = self.batch_index

        return ActivationSlicer.get(self.activation()).extract(index)

    def activation(self) -> ModelLayerLocation:
        return self[ModelLocation.ACTIVATION]

    def transformer_activation(self) -> str:
        return get_act_name(TRANSFORMER_ACT_NAME[self.activation()], layer=self.layer())

    def layer(self) -> int | None:
        if "layer" in self.index_names:
            layer = self.index[self.index_names.index("layer")]
            if not isinstance(layer, str) and not isinstance(layer, Sequence):
                return layer
        return None

    def set_batch_index(self, batch_index: int):
        self.batch_index = batch_index

    def __getitem__(self, item):
        return self.index[self.index_names.index(item)]
