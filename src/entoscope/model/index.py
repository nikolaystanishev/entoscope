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

from entoscope.model.config import (
    ModelLayerLocation,
    ModelLocation,
    ModelLocationIndex,
    ModelLocatorConfig,
    make_df_from_ranges,
)
from entoscope.model.slicer import ActivationSlicer
from entoscope.util.collections import broadcast_length, dataframe_to_list, zip_broadcast

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
    """One side of an index row: its location values by name, read off the locator config ``config``.

    ``slices`` turns the values into the slice of the activation tensor they address, and
    ``transformer_activation`` names the hook that tensor is cached under. A list value addresses
    several entries of its axis at once; a location the row does not name is the whole axis.
    """

    def __init__(self, config: T, index_names: list[str], index: list[int | list[int] | str | ModelLayerLocation]):
        self.config = config
        self.index_names = deepcopy(index_names)
        self.index = index

    def _index_by_name(self, batch: int | slice = slice(None)) -> defaultdict[str, int | list[int] | slice]:
        index: defaultdict[str, int | list[int] | slice] = defaultdict(lambda: slice(None))
        index.update(
            {name: idx for name, idx in zip(self.index_names, self.index, strict=True) if isinstance(idx, str) is False}
        )
        index["batch"] = batch

        return index

    def slices(self, batch: int | slice = slice(None)) -> tuple[slice | int | list[int], ...]:
        """The element's slices into an activation, at ``batch`` along its batch axis.

        The batch row is an argument rather than state on the element: ``zip_broadcast`` hands the same
        element to every row a one-row index is paired with, so a row stored on it would be shared.
        """
        return ActivationSlicer.get(self.activation()).extract(self._index_by_name(batch))

    def remaining_axes(self) -> list[ModelLocation | str]:
        """Names of the activation axes that survive ``slices()``, batch excluded.

        An ``int`` index drops its axis while a ``slice`` or ``list`` keeps it. Two or more
        ``list`` indices trigger NumPy-style advanced indexing, which merges those axes into
        one whose name can no longer be attributed - that case raises instead.
        """
        slicer = ActivationSlicer.get(self.activation())
        slices = slicer.extract(self._index_by_name())
        if sum(isinstance(idx, list) for idx in slices) > 1:
            raise ValueError("Cannot name the remaining axes when more than one axis is indexed with a list")
        return [
            name for name, idx in zip(slicer.AXES, slices, strict=True) if name != "batch" and not isinstance(idx, int)
        ]

    def activation(self) -> ModelLayerLocation:
        """The row's activation, as a member whichever way the frame stores it (pandas 3 keeps a string)."""
        activation = self[ModelLocation.ACTIVATION]
        return activation if activation is None else ModelLayerLocation(activation)

    def transformer_activation(self) -> str:
        return get_act_name(TRANSFORMER_ACT_NAME[self.activation()], layer=self.layer())

    def layer(self) -> int | None:
        layer = self.get(ModelLocation.LAYER)
        if not isinstance(layer, str) and not isinstance(layer, Sequence):
            return layer
        return None

    def get(self, location: ModelLocation | str) -> int | list[int] | str | None:
        return self[location] if location in self.index_names else None

    def __getitem__(self, location: ModelLocation | str):
        return self.index[self.index_names.index(location)]


class SingleIndex(Generic[T]):
    """One row of an ``Index``: where to read, where to write, and which result row it fills.

    ``output`` is that row's position in the index - the key a collector files its cell under and the
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


class Index(Generic[T]):
    """The rows an operation iterates, one ``SingleIndex`` each.

    ``source`` is a parsed locator; patching adds the ``target`` it writes into, paired with the source
    row by row - so the two have as many rows, or one of them a single row, which pairs with every row of
    the other. ``to_df`` is the frame ``index.csv`` holds, its columns prefixed ``source_`` and ``target_``,
    and ``layout_df`` codes the axes a result page lays the rows out on.
    """

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

        if self.target is None:
            self._length = len(self.source.df)
        else:
            try:
                self._length = broadcast_length(len(self.source.df), len(self.target.df))
            except ValueError:
                raise ValueError(
                    f"A source of {len(self.source.df)} rows cannot be paired with a target of {len(self.target.df)}: "
                    "they pair row by row, so they need as many rows, or one of them a single row"
                ) from None

        laid_out = self.target.df if self.target is not None and len(self.target.df) == len(self) else self.source.df
        if self.source_config.coordinates is not None and (
            self.target_config is None or self.target_config.coordinates is not None
        ):
            self.layout_df = make_df_from_ranges(["coordinate"], [len(laid_out)])
        else:
            axes = laid_out.drop(columns=ModelLocation.ACTIVATION, errors="ignore")
            self.layout_df = pd.DataFrame(
                {column: pd.factorize(axes[column].astype(str))[0] for column in axes.columns}
            )

    def source_activation_names(self) -> set[str]:
        return self._activation_names(self.source_config, self.source)

    def target_activation_names(self) -> set[str]:
        return self._activation_names(self.target_config, self.target)

    @staticmethod
    def _activation_names(config: T | None, index: ModelLocationIndex | None) -> set[str]:
        if config is None or index is None:
            return set()
        columns = index.df.columns.to_list()
        elements = (IndexElement(config, columns, row) for row in dataframe_to_list(index.df))
        return {element.transformer_activation() for element in elements if element.activation() is not None}

    def to_df(self) -> pd.DataFrame:
        frames = [self._paired(self.source.df).add_prefix("source_")]
        if self.target is not None:
            frames.append(self._paired(self.target.df).add_prefix("target_"))

        return pd.concat(frames, axis=1)

    def _paired(self, frame: pd.DataFrame) -> pd.DataFrame:
        """``frame``'s rows in the order they pair, a single row repeated along every row of the other side."""
        rows = [0] * len(self) if len(frame) == 1 else list(range(len(frame)))
        return frame.iloc[rows].reset_index(drop=True)

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
                zip_broadcast(dataframe_to_list(self.source.df), dataframe_to_list(self.target.df))
            )
        )

    def __len__(self) -> int:
        return self._length
