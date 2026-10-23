from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Generic, TypeVar

import pandas as pd
import torch

from entoscope.model.activation import transformer_activation, writable
from entoscope.model.config import (
    POSITION_AXES,
    ModelAxis,
    ModelLayerLocation,
    ModelLocationIndex,
    ModelLocatorConfig,
    make_df_from_ranges,
)
from entoscope.model.slicer import ActivationSlicer
from entoscope.util.collections import dataframe_to_list, zip_broadcast

T = TypeVar("T", bound="ModelLocatorConfig")


class IndexElement(Generic[T]):
    """One side of an index row: its values by axis, read off a locator config.

    ``slices`` turns the values into the slice of the activation tensor they address, and
    ``transformer_activation`` is the transformer activation name that tensor is cached under. A list value addresses
    several entries of its axis at once, lists on several axes every combination of their entries, and
    an axis the row does not name is read whole.

    Args:
        config: The locator the row was resolved from.
        index_names: The row's keys, ``ModelAxis`` values.
        index: The row's values, one per key.
    """

    def __init__(self, config: T, index_names: list[str], index: list[int | list[int] | str | ModelLayerLocation]):
        self.config = config
        self.index_names = index_names
        self.index = index

    def _index_by_name(self, batch: int | slice = slice(None)) -> defaultdict[str, int | list[int] | slice]:
        index: defaultdict[str, int | list[int] | slice] = defaultdict(lambda: slice(None))
        index.update(
            {name: idx for name, idx in zip(self.index_names, self.index, strict=True) if isinstance(idx, str) is False}
        )
        index["batch"] = batch

        return index

    def slices(self, batch: int | slice = slice(None)) -> tuple[slice | int | list[int], ...]:
        """Slice the element's activation.

        The batch row is an argument rather than state on the element: ``zip_broadcast`` hands the same
        element to every row a one-row index is paired with, so a row stored on it would be shared.
        The slices are read and written with ``read_slices`` and ``write_slices``: indexing a tensor with
        them directly would pair the lists of several axes.

        Args:
            batch: The batch rows to read.

        Returns:
            One index per axis of the activation tensor, in the order the tensor holds its axes.
        """
        return ActivationSlicer.get(self.activation()).extract(self._index_by_name(batch))

    def read_slices(self, tensor: torch.Tensor, batch: int | slice = slice(None)) -> torch.Tensor:
        """Read the element's block of an activation of its kind.

        Args:
            tensor: The activation.
            batch: The batch rows to read.

        Returns:
            The block of every combination of the element's values, every axis an int does not drop kept
            where the tensor holds it.
        """
        return tensor[self._block(tensor, self.slices(batch))]

    def read_unit_slices(self, weight: torch.Tensor) -> torch.Tensor:
        """Read the element's units off a weight that holds one ``d_model`` vector per unit of its activation.

        The weight's leading axes are the activation's other than ``batch`` and its positions, as
        ``model_weight`` lays it out.

        Args:
            weight: The weight.

        Returns:
            The block of the units the element names, ``d_model`` last.
        """
        axes = ActivationSlicer.get(self.activation()).AXES
        units = [
            value
            for axis, value in zip(axes, self.slices(), strict=True)
            if axis != "batch" and axis not in POSITION_AXES
        ]
        return weight[self._block(weight, units)]

    def write_slices(self, tensor: torch.Tensor, value: torch.Tensor, batch: int | slice = slice(None)) -> torch.Tensor:
        """Write into the block ``read_slices`` reads, in the tensor ``writable`` picks.

        The value is cast to the tensor's dtype and device first, as assigning into a slice would: a write
        through index tensors takes only an exact match, and a constant such as ``value_replace``'s is
        float32 whatever the model's dtype.

        Args:
            tensor: The activation to write into.
            value: The block, or anything that broadcasts to it.
            batch: The batch rows to write.

        Returns:
            The tensor written into: ``tensor`` itself, or a copy of a sublayer's input.
        """
        tensor = writable(self.transformer_activation(), tensor)
        tensor[self._block(tensor, self.slices(batch))] = torch.as_tensor(
            value, dtype=tensor.dtype, device=tensor.device
        )
        return tensor

    @staticmethod
    def _block(tensor: torch.Tensor, slices: Sequence[slice | int | list[int]]) -> tuple[int | torch.Tensor, ...]:
        """Index the block of every combination of the values on each of the tensor's leading axes.

        torch pairs the lists of several axes indexed together, so every axis an int does not drop is
        indexed by a tensor laid along its own axis of the block instead, and the tensors broadcast to every
        combination. An axis left after ``slices`` is kept whole, after the block's.

        Args:
            tensor: The tensor to index.
            slices: One value per leading axis of the tensor.

        Returns:
            The index, which reads and writes the block alike.
        """
        kept = sum(not isinstance(value, int) for value in slices)
        index: list[int | torch.Tensor] = []
        block_axis = 0
        for dim, value in enumerate(slices):
            if isinstance(value, int):
                index.append(value)
                continue
            if isinstance(value, slice):
                values = torch.arange(tensor.shape[dim], device=tensor.device)[value]
            else:
                values = torch.tensor(value, dtype=torch.long, device=tensor.device)
            shape = [1] * kept
            shape[block_axis] = -1
            index.append(values.view(shape))
            block_axis += 1
        return tuple(index)

    def remaining_axes(self) -> list[ModelAxis | str]:
        """Name the activation axes that survive ``slices()``, batch excluded.

        An ``int`` index drops its axis while a ``slice`` or ``list`` keeps it.

        Returns:
            The surviving axes, in tensor order.
        """
        slicer = ActivationSlicer.get(self.activation())
        slices = slicer.extract(self._index_by_name())
        return [
            name for name, idx in zip(slicer.AXES, slices, strict=True) if name != "batch" and not isinstance(idx, int)
        ]

    def activation(self) -> ModelLayerLocation:
        """Read the row's activation, as a member whichever way the frame stores it (pandas 3 keeps a string).

        Returns:
            The activation, or ``None`` when the locator names none.
        """
        activation = self[ModelAxis.ACTIVATION]
        return activation if activation is None else ModelLayerLocation(activation)

    def transformer_activation(self) -> str:
        """Name the row's transformer activation, the name its tensor is cached under.

        Returns:
            The transformer activation name, such as ``blocks.3.hook_resid_post``.
        """
        return transformer_activation(self.activation(), self.layer())

    def layer(self) -> int | None:
        """Read the row's layer.

        Returns:
            The layer, or ``None`` when the row names no single layer.
        """
        layer = self.get(ModelAxis.LAYER)
        if not isinstance(layer, str) and not isinstance(layer, Sequence):
            return layer
        return None

    def get(self, location: ModelAxis | str) -> int | list[int] | str | None:
        """Read the row's value on one axis.

        Args:
            location: The axis, or ``activation``.

        Returns:
            The value, or ``None`` when the row does not name the axis.
        """
        return self[location] if location in self.index_names else None

    def __getitem__(self, location: ModelAxis | str):
        return self.index[self.index_names.index(location)]


class SingleIndex(Generic[T]):
    """One row of an ``Index``: where to read, where to write, and which result row it fills.

    Positions are what ``Index`` iteration already hands out in order, so a run needs no coordinate of
    its own to find its row.

    Args:
        output_position: The row's position in the index.
        source: Where the row reads.
        target: Where the row writes, for patching.

    Attributes:
        output: The row's position in the index, the key a collector files its cell under and the row
            it lands on in the result frame.
    """

    def __init__(self, output_position: int, source: IndexElement[T], target: IndexElement | None = None):
        self.output = output_position
        self._source = source
        self._target = target

    @property
    def source(self) -> IndexElement[T]:
        """Where the row reads."""
        return self._source

    @property
    def target(self) -> IndexElement[T]:
        """Where the row writes. Only a patching index has one."""
        assert self._target is not None
        return self._target


class Index(Generic[T]):
    """The rows an operation iterates, one ``SingleIndex`` each, built once as the index is.

    Patching adds a target to write into, paired with the source row by row, so the two have as many
    rows, or one of them a single row, which pairs with every row of the other.

    Args:
        source_config: The locator the source was resolved from.
        source: The resolved source.
        target_config: The locator the target was resolved from, for patching.
        target: The resolved target, for patching.

    Attributes:
        layout_df: One integer code per row on every axis a result page lays the rows out on.
        layout_prefixes: The result columns' prefixes, the side ``layout_df`` was built from first, so a
            page labels its axes with the side that spans them.

    Raises:
        ValueError: If the source and the target cannot be paired row by row.
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

        source_columns = self.source.df.columns.to_list()
        sources = [
            IndexElement[T](self.source_config, source_columns, row) for row in dataframe_to_list(self.source.df)
        ]
        if self.target is None or self.target_config is None:
            self._rows = [SingleIndex[T](position, source_element) for position, source_element in enumerate(sources)]
        else:
            target_columns = self.target.df.columns.to_list()
            targets = [
                IndexElement[T](self.target_config, target_columns, row) for row in dataframe_to_list(self.target.df)
            ]
            try:
                pairs = zip_broadcast(sources, targets)
            except ValueError:
                raise ValueError(
                    f"A source of {len(self.source.df)} rows cannot be paired with a target of {len(self.target.df)}: "
                    "they pair row by row, so they need as many rows, or one of them a single row"
                ) from None
            self._rows = [
                SingleIndex[T](position, source_element, target_element)
                for position, (source_element, target_element) in enumerate(pairs)
            ]

        target_laid_out = self.target is not None and len(self.target.df) == len(self)
        laid_out = self.target.df if target_laid_out else self.source.df
        if self.target is None:
            self.layout_prefixes: tuple[str, ...] = ("source_",)
        else:
            self.layout_prefixes = ("target_", "source_") if target_laid_out else ("source_", "target_")
        if self.source_config.coordinates is not None and (
            self.target_config is None or self.target_config.coordinates is not None
        ):
            self.layout_df = make_df_from_ranges(["coordinate"], [len(laid_out)])
        else:
            axes = laid_out.drop(columns=ModelAxis.ACTIVATION, errors="ignore")
            self.layout_df = pd.DataFrame(
                {column: pd.factorize(axes[column].astype(str))[0] for column in axes.columns}
            )

    def activation_names(self) -> list[str]:
        """Name the transformer activations the rows read or write.

        A source without an activation, a value a patch writes, reads none.

        Returns:
            The transformer activation names, sorted.
        """
        source_activation_names = {
            row.source.transformer_activation() for row in self if row.source.activation() is not None
        }
        return sorted(source_activation_names.union(self.target_activation_names()))

    def target_activation_names(self) -> list[str]:
        """Name the transformer activations the target rows write.

        Returns:
            The transformer activation names, sorted, none without a target.
        """
        if self.target is None or self.target_config is None:
            return []
        return sorted({row.target.transformer_activation() for row in self if row.target.activation() is not None})

    def to_df(self) -> pd.DataFrame:
        """Lay the index out as ``index.csv`` holds it.

        Returns:
            One row per index row, the source's columns prefixed ``source_`` and the target's
            ``target_``, a single-row side repeated along the other.
        """
        frames = [self._paired(self.source.df).add_prefix("source_")]
        if self.target is not None:
            frames.append(self._paired(self.target.df).add_prefix("target_"))

        return pd.concat(frames, axis=1)

    def _paired(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Order one side's rows as they pair.

        Args:
            frame: One side of the index.

        Returns:
            The rows, a single row repeated along every row of the other side.
        """
        rows = [0] * len(self) if len(frame) == 1 else list(range(len(frame)))
        return frame.iloc[rows].reset_index(drop=True)

    def to_csv(self, path: str):
        """Write ``to_df()`` as a CSV file.

        Args:
            path: The file to write.
        """
        self.to_df().to_csv(path, index=False)

    def __iter__(self) -> Iterator[SingleIndex[T]]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)
