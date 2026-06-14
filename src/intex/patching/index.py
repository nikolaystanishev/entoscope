from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy

import pandas as pd
import torch
from transformer_lens.utilities import get_act_name

from intex.model.config import ACTIVATION_NAME_COLUMN, ModelLayerLocation, ModelLocationIndex, make_df_from_ranges
from intex.operation.index import Index
from intex.patching.slicer import ActivationSlicer
from intex.util.collections import dataframe_to_list, zip_fill


class IndexElement:
    def __init__(self, index_names: list[str], index: list[int | list[int] | str]):
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
        return ModelLayerLocation(self[ACTIVATION_NAME_COLUMN])

    def transformer_activation(self) -> str:
        layer_idx = self.layer()
        return get_act_name(self.activation().replace("{layer}", f"{layer_idx}"), layer=layer_idx)

    def layer(self) -> int | None:
        assert "layer" in self.index_names or "layer_none" in self.index_names
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
        source_index_names: list[str],
        target_index_names: list[str],
        source_index: list[int],
        target_index: list[int],
        output_index: list[int],
    ):
        self.source = IndexElement(source_index_names, source_index)
        self.target = IndexElement(target_index_names, target_index)
        self.output = tuple(output_index)
        self.batch_index = None

    def set_batch_index(self, batch_index: int) -> "SinglePatchingIndex":
        self.batch_index = batch_index
        self.source.set_batch_index(batch_index)
        self.target.set_batch_index(batch_index)

        return self


class PatchingIndex(Index):
    def __init__(self, source_index: ModelLocationIndex, target_index: ModelLocationIndex):
        self.source_index = source_index
        self.target_index = target_index

        output_index_df = (
            self.target_index.df if len(self.target_index.df) >= len(self.source_index.df) else self.source_index.df
        )
        coordinate_df = output_index_df[[col for col in output_index_df.columns if col != ACTIVATION_NAME_COLUMN]]
        self.index_axis_range = coordinate_df.apply(lambda col: len(col.astype(str).unique())).to_list()
        self.output_index_df = make_df_from_ranges(self.index_axis_range, coordinate_df.columns.to_list())

    def prepare_output_placeholder(self, device: str | torch.device = "cpu") -> torch.Tensor:
        return torch.zeros(self.index_axis_range, device=device)

    def to_csv(self, path: str):
        combined_df = pd.concat(
            [self.source_index.df.add_prefix("source_"), self.target_index.df.add_prefix("target_")], axis=1
        )
        combined_df.to_csv(path, index=False)

    def __iter__(self):
        return map(
            lambda el: SinglePatchingIndex(
                self.source_index.df.columns.to_list(), self.target_index.df.columns.to_list(), *el
            ),
            zip_fill(
                dataframe_to_list(self.source_index.df),
                dataframe_to_list(self.target_index.df),
                dataframe_to_list(self.output_index_df),
            ),
        )

    def __len__(self) -> int:
        return len(self.output_index_df)
