from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from typing import Generic, TypeVar

import pandas as pd

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
    def __init__(self, config: T, index_names: list[str], index: list[int | list[int] | str | ModelLayerLocation]):
        self.config = config
        self.index_names = deepcopy(index_names)
        self.index = index
        self.batch_index = None

    def _index_by_name(self) -> defaultdict[str, int | list[int] | slice]:
        index: defaultdict[str, int | list[int] | slice] = defaultdict(lambda: slice(None))
        index.update(
            {name: idx for name, idx in zip(self.index_names, self.index, strict=True) if isinstance(idx, str) is False}
        )

        if self.batch_index is not None:
            index["batch"] = self.batch_index

        return index

    def slices(self) -> tuple[slice | int | list[int], ...]:
        return ActivationSlicer.get(self.activation()).extract(self._index_by_name())

    def remaining_axes(self) -> list[ModelLocation | str]:
        """Names of the activation axes that survive ``slices()``, batch excluded.

        An ``int`` index drops its axis while a ``slice`` or ``list`` keeps it. Two or more
        ``list`` indices trigger NumPy-style advanced indexing, which merges those axes into
        one whose name can no longer be attributed — that case raises instead.
        """
        slicer = ActivationSlicer.get(self.activation())
        slices = slicer.extract(self._index_by_name())
        if sum(isinstance(idx, list) for idx in slices) > 1:
            raise ValueError("Cannot name the remaining axes when more than one axis is indexed with a list")
        return [
            name for name, idx in zip(slicer.AXES, slices, strict=True) if name != "batch" and not isinstance(idx, int)
        ]

    def activation(self) -> ModelLayerLocation:
        return self[ModelLocation.ACTIVATION]

    def transformer_activation(self) -> str:
        return get_act_name(TRANSFORMER_ACT_NAME[self.activation()], layer=self.layer())

    def layer(self) -> int | None:
        layer = self.get(ModelLocation.LAYER)
        if not isinstance(layer, str) and not isinstance(layer, Sequence):
            return layer
        return None

    def set_batch_index(self, batch_index: int):
        self.batch_index = batch_index

    def get(self, location: ModelLocation | str) -> int | list[int] | str | None:
        return self[location] if location in self.index_names else None

    def __getitem__(self, location: ModelLocation | str):
        return self.index[self.index_names.index(location)]


class SingleIndex(Generic[T]):
    """One row of an ``Index``: where to read, where to write, and which result row it fills.

    ``output`` is that row's position in the index — the key a collector files its cell under and the
    row it lands on in the result frame. Positions are what ``Index`` iteration already hands out in
    order, so a run needs no coordinate of its own to find its row.
    """

    def __init__(self, output_position: int, source: IndexElement[T], target: IndexElement | None = None):
        self.output = output_position
        self._source = source
        self._target = target

    @property
    def source(self) -> IndexElement[T]:
        return self._source

    @property
    def target(self) -> IndexElement[T]:
        assert self._target is not None
        return self._target

    def set_batch_index(self, batch_index: int) -> SingleIndex[T]:
        self.source.set_batch_index(batch_index)
        if self._target is not None:
            self._target.set_batch_index(batch_index)

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
            self.output_df = make_df_from_ranges(["coordinate"], [len(output_df)])
        else:
            # Keep one output row per index row (the df may be a sparse, correlated set rather than
            # a full grid), mapping each column's values to dense first-appearance codes.
            coordinate_df = output_df[[col for col in output_df.columns if col != ModelLocation.ACTIVATION]]
            factorized = {col: pd.factorize(coordinate_df[col].astype(str)) for col in coordinate_df.columns}
            self.output_df = pd.DataFrame(
                {col: codes for col, (codes, _) in factorized.items()}, index=coordinate_df.index
            )

    def source_activation_names(self) -> set[str]:
        columns = self.source.df.columns.to_list()
        elements = (IndexElement(self.source_config, columns, row) for row in dataframe_to_list(self.source.df))
        return {element.transformer_activation() for element in elements if element.activation() is not None}

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
                SingleIndex[T](position, IndexElement[T](self.source_config, source_columns, source_row))
                for position, source_row in enumerate(dataframe_to_list(self.source.df))
            )

        target_columns = self.target.df.columns.to_list()
        return (
            SingleIndex[T](
                position,
                IndexElement[T](self.source_config, source_columns, source_row),
                IndexElement[T](self.target_config, target_columns, target_row),
            )
            for position, (source_row, target_row) in enumerate(
                zip_fill(dataframe_to_list(self.source.df), dataframe_to_list(self.target.df))
            )
        )

    def __len__(self) -> int:
        return len(self.output_df)
