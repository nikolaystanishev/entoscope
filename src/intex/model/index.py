from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from typing import Generic, TypeVar

import pandas as pd
import torch

try:
    from transformer_lens.utilities import get_act_name
except ImportError:  # transformer-lens 2.x (Intel macOS fallback)
    from transformer_lens.utils import get_act_name

from intex.model.config import (
    ModelLayerLocation,
    ModelLocation,
    ModelLocationIndex,
    ModelLocatorConfig,
    make_df_from_ranges,
)
from intex.model.slicer import ActivationSlicer
from intex.util.collections import dataframe_to_list, zip_fill

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

T = TypeVar("T", bound="ModelLocatorConfig")


class IndexElement(Generic[T]):
    def __init__(self, config: T, index_names: list[str], index: list[int | list[int] | str]):
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


class SingleIndex(Generic[T]):
    def __init__(self, source: IndexElement[T], output_index: list[int], target: IndexElement | None = None):
        self._source = source
        self._target = target
        self.output = tuple(output_index)

    @property
    def source(self) -> IndexElement[T]:
        return self._source

    @property
    def target(self) -> IndexElement[T]:
        assert self._target is not None
        return self._target

    def set_batch_index(self, batch_index: int) -> SingleIndex[T]:
        self.source.set_batch_index(batch_index)
        if self.target is not None:
            self.target.set_batch_index(batch_index)

        return self


class Index(Generic[T]):
    def __init__(
        self,
        source_config: T,
        source: ModelLocationIndex,
        target_config: T | None = None,
        target: ModelLocationIndex | None = None,
    ):
        self.source_config = source_config
        self.source = source
        self.target_config = target_config
        self.target = target

        output_df = (
            self.target.df if self.target is not None and len(self.target.df) >= len(self.source.df) else self.source.df
        )
        if self.source_config.coordinates is not None and (
            self.target_config is None or self.target_config.coordinates is not None
        ):
            self.axis_range = [len(output_df)]
            self.output_df = make_df_from_ranges(["coordinate"], self.axis_range)
        else:
            # Keep one output row per index row (the df may be a sparse, correlated set rather than
            # a full grid), mapping each column's values to dense first-appearance codes.
            coordinate_df = output_df[[col for col in output_df.columns if col != ModelLocation.ACTIVATION]]
            factorized = {col: pd.factorize(coordinate_df[col].astype(str)) for col in coordinate_df.columns}
            self.axis_range = [len(uniques) for _, uniques in factorized.values()]
            self.output_df = pd.DataFrame(
                {col: codes for col, (codes, _) in factorized.items()}, index=coordinate_df.index
            )

    def prepare_output_placeholder(self, device: str | torch.device = "cpu") -> torch.Tensor:
        return torch.zeros(self.axis_range, device=device)

    def to_df(self) -> pd.DataFrame:
        dfs = [self.source.df.add_prefix("source_")]
        if self.target is not None:
            dfs.append(self.target.df.add_prefix("target_"))
        dfs.append(self.output_df.add_prefix("index_"))

        return pd.concat(dfs, axis=1)

    def to_csv(self, path: str):
        self.to_df().to_csv(path, index=False)

    def __iter__(self):
        source_columns = self.source.df.columns.to_list()

        if self.target is None or self.target_config is None:
            return (
                SingleIndex[T](IndexElement[T](self.source_config, source_columns, source_row), output_row)
                for source_row, output_row in zip(
                    dataframe_to_list(self.source.df), dataframe_to_list(self.output_df), strict=True
                )
            )

        target_columns = self.target.df.columns.to_list()
        return (
            SingleIndex[T](
                IndexElement[T](self.source_config, source_columns, source_row),
                output_row,
                IndexElement[T](self.target_config, target_columns, target_row),
            )
            for source_row, target_row, output_row in zip_fill(
                dataframe_to_list(self.source.df), dataframe_to_list(self.target.df), dataframe_to_list(self.output_df)
            )
        )

    def __len__(self) -> int:
        return len(self.output_df)
