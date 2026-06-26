from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING

import pandas as pd
import torch
from transformer_lens import HookedTransformer
from transformer_lens.utilities import get_act_name

from intex.model.config import ACTIVATION_NAME_COLUMN, ModelLayerLocation, make_df_from_ranges
from intex.operation.index import Index
from intex.patching.config import PatchingElementConfig, SinglePatchingConfig
from intex.patching.slicer import ActivationSlicer
from intex.util.collections import dataframe_to_list, zip_fill

if TYPE_CHECKING:
    from intex.patching.input import PatchingInput

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
    def __init__(self, config: PatchingElementConfig, index_names: list[str], index: list[int | list[int] | str]):
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
        return self[ACTIVATION_NAME_COLUMN]

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

    def get(self, item, default=None):
        if item in self.index_names:
            return self[item]
        return default


class SinglePatchingIndex:
    def __init__(
        self,
        config: SinglePatchingConfig,
        source_index_names: list[str],
        target_index_names: list[str],
        source_index: list[int],
        target_index: list[int],
        output_index: list[int],
    ):
        self.source = IndexElement(config.source, source_index_names, source_index)
        self.target = IndexElement(config.target, target_index_names, target_index)
        self.output = tuple(output_index)

    def set_batch_index(self, batch_index: int) -> SinglePatchingIndex:
        self.source.set_batch_index(batch_index)
        self.target.set_batch_index(batch_index)

        return self


class PatchingIndex(Index):
    def __init__(self, config: SinglePatchingConfig, model: HookedTransformer, patching_input: PatchingInput):
        self.config = config
        self.source = config.source.parse(model, patching_input.source_tokens)
        self.target = config.target.parse(model, patching_input.target_tokens)

        output_df = self.target.df if len(self.target.df) >= len(self.source.df) else self.source.df
        coordinate_df = output_df[[col for col in output_df.columns if col != ACTIVATION_NAME_COLUMN]]
        self.axis_range = coordinate_df.apply(lambda col: len(col.astype(str).unique())).to_list()
        self.output_df = make_df_from_ranges(coordinate_df.columns.to_list(), self.axis_range)

    def prepare_output_placeholder(self, device: str | torch.device = "cpu") -> torch.Tensor:
        return torch.zeros(self.axis_range, device=device)

    def to_csv(self, path: str):
        pd.concat(
            [
                self.source.df.add_prefix("source_"),
                self.target.df.add_prefix("target_"),
                self.output_df.add_prefix("index_"),
            ],
            axis=1,
        ).to_csv(path, index=False)

    def __iter__(self):
        return map(
            lambda el: SinglePatchingIndex(
                self.config, self.source.df.columns.to_list(), self.target.df.columns.to_list(), *el
            ),
            zip_fill(
                dataframe_to_list(self.source.df), dataframe_to_list(self.target.df), dataframe_to_list(self.output_df)
            ),
        )

    def __len__(self) -> int:
        return len(self.output_df)
