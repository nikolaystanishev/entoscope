import dataclasses
import itertools
import sys
import typing
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

import numpy as np
import pandas as pd
from pandas._typing import Axes

from entoscope.model.base import Model
from entoscope.util.collections import range_combinations
from entoscope.util.data import from_dict


class ModelAxis(StrEnum):
    """The axes an index row names its location on, and the row's activation.

    ``layer``, ``head``, ``neuron`` and ``position`` are axes a locator can enumerate. ``activation`` is
    not an axis but a key of every row: it is set once on the locator (or on each coordinate) and stamped
    on every row.
    """

    ACTIVATION = "activation"
    LAYER = "layer"
    HEAD = "head"
    NEURON = "neuron"
    POSITION = "position"


class ModelLayerLocation(StrEnum):
    """The activations a locator can name, valued as a config file spells them.

    Each maps onto a ``transformer_lens`` hook through ``TRANSFORMER_ACT_NAME``, and the comment beside a
    member is its tensor's axes. ``embed`` is not inside a block, so a locator reading it names no layer.
    """

    EMBEDDING = "embed"  # [batch, position, neuron]
    RESIDUAL_PRE = "resid.pre"  # [batch, position, neuron]
    RESIDUAL_MID = "resid.mid"  # [batch, position, neuron]
    RESIDUAL_POST = "resid.post"  # [batch, position, neuron]
    ATTENTION_Q = "attn.q"  # [batch, position, head, neuron]
    ATTENTION_K = "attn.k"  # [batch, position, head, neuron]
    ATTENTION_V = "attn.v"  # [batch, position, head, neuron]
    ATTENTION_PATTERN = "attn.pattern"  # [batch, head, dest_position /query/, src_position /key/]
    ATTENTION_Z = "attn.z"  # [batch, position, head, neuron]
    ATTENTION_OUT = "attn.out"  # [batch, position, neuron]
    MLP_PRE = "mlp.pre"  # [batch, position, neuron]
    MLP_PRE_LINEAR = "mlp.pre_linear"  # [batch, position, neuron]
    MLP_POST = "mlp.post"  # [batch, position, neuron]
    MLP_OUT = "mlp.out"  # [batch, position, neuron]


@dataclass
class ModelLocationIndex:
    """A locator resolved against a model and a prompt length.

    Attributes:
        df: One row per location, one column per axis the locator names and one for the activation.
    """

    df: pd.DataFrame = field(default_factory=pd.DataFrame)


MODEL_LOCATION_MAPPER: dict[ModelAxis, Callable] = {
    ModelAxis.LAYER: lambda model, _1, _2: model.cfg.n_layers,
    ModelAxis.HEAD: lambda model, layer_location, _: (
        model.cfg.n_heads
        if layer_location not in {ModelLayerLocation.ATTENTION_K, ModelLayerLocation.ATTENTION_V}
        else model.cfg.n_key_value_heads or model.cfg.n_heads
    ),
    ModelAxis.NEURON: (
        lambda model, layer_location, _: {
            ModelLayerLocation.EMBEDDING: model.cfg.d_model,
            ModelLayerLocation.RESIDUAL_PRE: model.cfg.d_model,
            ModelLayerLocation.RESIDUAL_MID: model.cfg.d_model,
            ModelLayerLocation.RESIDUAL_POST: model.cfg.d_model,
            ModelLayerLocation.ATTENTION_Q: model.cfg.d_head,
            ModelLayerLocation.ATTENTION_K: model.cfg.d_head,
            ModelLayerLocation.ATTENTION_V: model.cfg.d_head,
            ModelLayerLocation.ATTENTION_PATTERN: 0,
            ModelLayerLocation.ATTENTION_Z: model.cfg.d_head,
            ModelLayerLocation.ATTENTION_OUT: model.cfg.d_model,
            ModelLayerLocation.MLP_PRE: model.cfg.d_mlp,
            ModelLayerLocation.MLP_PRE_LINEAR: model.cfg.d_mlp,
            ModelLayerLocation.MLP_POST: model.cfg.d_mlp,
            ModelLayerLocation.MLP_OUT: model.cfg.d_model,
        }[layer_location]
    ),
    ModelAxis.POSITION: lambda _1, _2, prompt_size: prompt_size,
}


def count_from_end(value: int, max_value: int | None) -> int:
    """Resolve a value on an axis, a negative one counting from the axis's end.

    Args:
        value: A value on the axis.
        max_value: The axis's size. ``None`` is a prompt length that varies, so the value stays
            symbolic and torch resolves a negative one against each prompt's own cache at slice time.

    Returns:
        The value, counted from the start when the axis's size is known.
    """
    return value if value >= 0 or max_value is None else max_value + value


def make_df_from_ranges(column_names: Axes, column_ranges: Sequence[int | Sequence[int]]) -> pd.DataFrame:
    """Enumerate every combination of the axes' values.

    Args:
        column_names: The axes.
        column_ranges: Each axis's values, or its size for every value from 0.

    Returns:
        One row per combination, one column per axis.
    """
    return pd.DataFrame(
        data=itertools.product(
            *[axis_range if isinstance(axis_range, Sequence) else range(axis_range) for axis_range in column_ranges]
        ),
        columns=column_names,
    )


class IndexTransform(Protocol):
    """A config section that rewrites the index DataFrame in the parse pipeline."""

    def apply(self, index_df: pd.DataFrame, location_to_max_value: dict) -> pd.DataFrame:
        """Rewrite the index.

        Args:
            index_df: The index so far.
            location_to_max_value: The size of every axis the transform names, so a negative value
                counts from its end.

        Returns:
            The rewritten index.
        """
        ...


@dataclass
class Filter:
    """Keeps or drops index rows by a list of patterns.

    Each pattern maps an axis to a constraint: the keys of one pattern are AND-ed, the patterns are
    OR-ed, and ``{"not": <pattern>}`` drops the rows it matches instead. A constraint is one of:

    - an int, matching that value;
    - ``{"range": [start, end]}``, matching ``start`` to ``end`` with both ends included;
    - a list of the above, matching any of them.

    Negative numbers count from the end. The result is the union of the positive patterns
    (everything when there are none) minus the union of the ``not`` patterns.

    Attributes:
        inclusion_terms: The patterns a row is kept by, keyed by ``ModelAxis`` members.
        exclusion_terms: The patterns a row is dropped by, keyed by ``ModelAxis`` members.
    """

    inclusion_terms: list[dict[ModelAxis, int | dict | list]] = field(default_factory=list)
    exclusion_terms: list[dict[ModelAxis, int | dict | list]] = field(default_factory=list)

    @classmethod
    def from_config(cls, terms: list) -> "Filter":
        """Read a config file's filter.

        Args:
            terms: The patterns, ``{"not": <pattern>}`` for one that drops rows. A pattern's keys are
                ``ModelAxis`` members or their values.

        Returns:
            The filter.

        Raises:
            ValueError: If a term is not a non-empty mapping, or a key names no axis.
        """
        inclusion_terms: list[dict] = []
        exclusion_terms: list[dict] = []
        for term in terms:
            pattern = term["not"] if isinstance(term, dict) and set(term.keys()) == {"not"} else term
            if not isinstance(pattern, dict) or len(pattern) == 0:
                raise ValueError(f"Invalid filter term: {term!r}")
            pattern = {ModelAxis(location): constraint for location, constraint in pattern.items()}

            if isinstance(term, dict) and set(term.keys()) == {"not"}:
                exclusion_terms.append(pattern)
            else:
                inclusion_terms.append(pattern)

        return cls(inclusion_terms=inclusion_terms, exclusion_terms=exclusion_terms)

    def to_config(self) -> list:
        """Write the filter as a config file's list of terms, which ``from_config`` reads back.

        Returns:
            The inclusion patterns, then each exclusion pattern as ``{"not": <pattern>}``.
        """
        return [*self.inclusion_terms, *({"not": term} for term in self.exclusion_terms)]

    def included(self, location: ModelAxis, max_value: int | None) -> set[int]:
        """Find the values of one axis some inclusion pattern can keep.

        Args:
            location: The axis.
            max_value: The axis's size, ``None`` for a prompt length that varies.

        Returns:
            Every value of the axis when a pattern leaves it unconstrained, otherwise the values the
            patterns name.

        Raises:
            ValueError: If every value would be kept on an axis whose size varies with the prompt.
        """
        if not self.inclusion_terms or any(location not in term for term in self.inclusion_terms):
            if max_value is None:
                raise ValueError(
                    f"Cannot build a grid over '{location.value}' when the prompt length varies; "
                    "filter to explicit positions or pool over them"
                )
            return set(range(max_value))
        return set().union(*(self._resolve(term[location], location, max_value) for term in self.inclusion_terms))

    def excluded(self, location: ModelAxis, max_value: int | None) -> set[int]:
        """Find the values of one axis an exclusion pattern on that axis alone drops.

        Args:
            location: The axis.
            max_value: The axis's size, ``None`` for a prompt length that varies.

        Returns:
            The values dropped whatever the row's other axes hold.
        """
        return set().union(
            *(
                self._resolve(term[location], location, max_value)
                for term in self.exclusion_terms
                if set(term.keys()) == {location}
            )
        )

    def location_range(self, location: ModelAxis, max_value: int | None) -> list[int]:
        """Find the values of one axis worth enumerating before the filter is applied row by row.

        Args:
            location: The axis.
            max_value: The axis's size, ``None`` for a prompt length that varies.

        Returns:
            The included values minus the excluded ones, ascending.
        """
        return sorted(self.included(location, max_value) - self.excluded(location, max_value))

    def apply(self, index_df: pd.DataFrame, location_to_max_value: dict) -> pd.DataFrame:
        """Keep the rows matching an inclusion pattern and drop those matching an exclusion pattern.

        Args:
            index_df: The index.
            location_to_max_value: The size of every axis, so a negative value counts from its end.

        Returns:
            The filtered index.

        Raises:
            ValueError: If a pattern names an axis the index does not have, or holds an invalid
                constraint.
        """
        if len(self.inclusion_terms) > 0:
            index_df = index_df[
                np.any([self._match(index_df, term, location_to_max_value) for term in self.inclusion_terms], axis=0)
            ]
        if len(self.exclusion_terms) > 0:
            index_df = index_df[
                ~np.any([self._match(index_df, term, location_to_max_value) for term in self.exclusion_terms], axis=0)
            ]
        return index_df

    @classmethod
    def _match(cls, index_df: pd.DataFrame, term: dict, location_to_max_value: dict) -> np.ndarray:
        unknown = [name for name in term if name not in index_df.columns]
        if unknown:
            raise ValueError(f"Filter pattern references unknown locations {unknown}: {term!r}")
        return np.all(
            [
                index_df[name].isin(cls._resolve(constraint, name, location_to_max_value[name]))
                for name, constraint in term.items()
            ],
            axis=0,
        )

    @classmethod
    def _resolve(cls, constraint: int | dict | list, location: ModelAxis, max_value: int | None) -> set[int]:
        if isinstance(constraint, list):
            return set().union(*(cls._resolve(el, location, max_value) for el in constraint))
        if isinstance(constraint, int):
            return {count_from_end(constraint, max_value)}
        if isinstance(constraint, dict) and set(constraint.keys()) == {"range"}:
            start, end = constraint["range"]
            if max_value is None and (start < 0) != (end < 0):
                raise ValueError(
                    f"Cannot resolve range {constraint['range']} for '{location.value}' when the prompt "
                    "length varies; use positions on the same side (all from the start or all from the end)"
                )
            return set(range(count_from_end(start, max_value), count_from_end(end, max_value) + 1))
        raise ValueError(f"Invalid filter constraint for '{location}': {constraint!r}")


class Group(dict[ModelAxis, list[int] | str]):
    """Merges the listed values of an axis into one row that addresses them together.

    A mapping from axis to the values to merge: ``{head: [6, 9]}`` turns the rows of heads 6 and 9
    into a single row reading both heads, and ``"all"`` merges every value the axis has. The duplicate
    rows a merge leaves are dropped. Merge heads, neurons or positions: every layer is a hook of its
    own, so a row naming several layers names no activation a run can read.
    """

    def apply(self, index_df: pd.DataFrame, location_to_max_value: dict) -> pd.DataFrame:
        """Merge the listed values of every axis.

        Args:
            index_df: The index.
            location_to_max_value: The size of every axis, so a negative value counts from its end.

        Returns:
            The index with each merged group in one row.
        """
        for name, values in self.items():
            if values == "all":
                values = index_df[name].unique().tolist()
            else:
                values = [count_from_end(value, location_to_max_value[name]) for value in values]
            values_to_replace = {v: values for v in values}
            index_df[name] = index_df[name].apply((lambda mapping: lambda el: mapping.get(el, el))(values_to_replace))
        index_df = index_df.loc[index_df.astype(str).drop_duplicates().index]

        return index_df


class Incremental(dict[ModelAxis, list[int]]):
    """Follows every row with rows that add growing prefixes of the listed values to an axis.

    A mapping from axis to the values to add: ``{head: [6, 9]}`` turns the row of head 0 into rows for
    heads ``[0]``, ``[0, 6]`` and ``[0, 6, 9]``. An axis the index does not have yet is added, and then
    only the grown rows exist.
    """

    def apply(self, index_df: pd.DataFrame, location_to_max_value: dict) -> pd.DataFrame:
        """Grow every row along every listed axis.

        Args:
            index_df: The index.
            location_to_max_value: The size of every axis, so a negative value counts from its end.

        Returns:
            The index with the grown rows after each row they grow from.
        """
        for name, values in self.items():
            values = [count_from_end(value, location_to_max_value[name]) for value in values]
            new_rows = []
            for _, row in index_df.iterrows():
                row = row.astype(object)
                if name in row:
                    if not isinstance(row[name], list):
                        row[name] = [row[name]]
                    new_rows.append(row)
                else:
                    row[name] = []

                for i in range(1, len(values) + 1):
                    new_row = deepcopy(row)
                    new_row[name] = new_row[name] + values[:i]
                    new_rows.append(new_row)

            if len(new_rows) > 0:
                columns = list(new_rows[0].index)
                index_df = pd.DataFrame(
                    {col: pd.Series([row[col] for row in new_rows], dtype=object) for col in columns}
                )

        return index_df


@dataclass
class Combination:
    """Expands each row into one row per combination of the values it holds on each listed axis.

    Combinations run from one value up to ``max_elements``, so it follows a ``group``: after
    ``group: {head: all}`` over heads 6, 9 and 10, ``max_elements: 2`` gives the rows ``(6,)``,
    ``(9,)``, ``(10,)``, ``(6, 9)``, ``(6, 10)`` and ``(9, 10)``.

    Attributes:
        location: The axes to expand.
        max_elements: The most values a combination holds.
    """

    location: list[ModelAxis] = field(default_factory=list)
    max_elements: int = sys.maxsize

    def apply(self, index_df: pd.DataFrame, location_to_max_value: dict) -> pd.DataFrame:
        """Expand every row along every listed axis.

        Args:
            index_df: The index, holding a list of values on every listed axis.
            location_to_max_value: Unused, as the values are resolved already.

        Returns:
            The index with each row replaced by its combinations.
        """
        for location in self.location:
            new_rows = []
            for _, row in index_df.iterrows():
                values = row[location]
                value_combinations = range_combinations(values, 1, min(len(values), self.max_elements))
                for el in value_combinations:
                    new_row = deepcopy(row)
                    new_row[location] = el
                    new_rows.append(new_row)
            if len(new_rows) > 0:
                index_df = pd.DataFrame(new_rows)

        return index_df


class Coordinates(list[dict[ModelAxis, int | list[int] | ModelLayerLocation]]):
    """Explicit coordinates, each a mapping from ``ModelAxis`` to a value.

    A value is an int or a list of ints read together, and a coordinate's ``activation`` a
    ``ModelLayerLocation``.
    """

    @classmethod
    def from_config(cls, coordinates: list) -> "Coordinates":
        """Read a config file's coordinates.

        Args:
            coordinates: The coordinates, keyed by ``ModelAxis`` members or their values, with an
                ``activation`` a ``ModelLayerLocation`` member or its value.

        Returns:
            The coordinates, keyed by ``ModelAxis`` members.

        Raises:
            ValueError: If ``coordinates`` is not a list, a coordinate is not a non-empty mapping, or a
                key or an activation is not valid.
        """
        if not isinstance(coordinates, list):
            raise ValueError(f"Expected a list of coordinates, got {coordinates!r}")
        for coordinate in coordinates:
            if not isinstance(coordinate, dict) or len(coordinate) == 0:
                raise ValueError(f"Invalid coordinate: {coordinate!r}")

        return cls(
            {
                ModelAxis(key): (ModelLayerLocation(value) if ModelAxis(key) is ModelAxis.ACTIVATION else value)
                for key, value in coordinate.items()
            }
            for coordinate in coordinates
        )

    def to_df(
        self,
        model: Model,
        n_positions: int | None,
        activation: ModelLayerLocation | None = None,
    ) -> pd.DataFrame:
        """Resolve the coordinates into an index, negative values counting from the end of their axis.

        Args:
            model: The model the axes are sized by.
            n_positions: The prompt's length, ``None`` when it varies.
            activation: The activation of a coordinate that names none.

        Returns:
            One row per coordinate, in order, one column per key.

        Raises:
            ValueError: If a value is neither an int nor a list of ints.
        """
        keys = sorted(list(self[0].keys()) if self else [], key=lambda key: key == ModelAxis.ACTIVATION)
        return pd.DataFrame(
            {
                key: pd.Series(
                    [self._resolve_entry(model, n_positions, coordinate, key, activation) for coordinate in self],
                    dtype=object,
                )
                for key in keys
            }
        )

    @classmethod
    def _resolve_entry(
        cls,
        model: Model,
        n_positions: int | None,
        coordinate: dict,
        key: ModelAxis,
        activation: ModelLayerLocation | None,
    ) -> int | list[int] | ModelLayerLocation:
        if key == ModelAxis.ACTIVATION:
            return coordinate[key]
        activation = coordinate.get(ModelAxis.ACTIVATION, activation)
        max_value = MODEL_LOCATION_MAPPER[key](model, activation, n_positions)
        return cls._resolve_value(coordinate[key], key, max_value)

    @classmethod
    def _resolve_value(cls, value: int | list[int], location: ModelAxis, max_value: int | None) -> int | list[int]:
        if isinstance(value, list):
            return [cls._resolve_value(el, location, max_value) for el in value]
        if isinstance(value, int):
            return count_from_end(value, max_value)
        raise ValueError(f"Invalid coordinate value for '{location}': {value!r}")


@dataclass(kw_only=True)
class ModelLocatorConfig:
    """Locates model activations, either declaratively or explicitly.

    Declaratively, ``location`` names the index axes and the full range of every axis is narrowed
    by ``filter`` and reshaped by the transforms. Explicitly, ``coordinates`` enumerates the exact
    points, every coordinate naming the same axes. The two modes are mutually exclusive. Negative
    values count from the end of their axis. A field built in code may hold its raw config form, such
    as a list for ``filter``, which is read as a config file's would be.

    Attributes:
        activation: The activation every row reads. Left unset when every coordinate names its own.
        location: The axes the index enumerates.
        filter: Which rows of the enumerated grid to keep.
        group: The values of an axis to merge into one row.
        incremental: The values of an axis to add to every row one at a time.
        combination: The axes to expand into every combination of their values.
        coordinates: The exact rows, instead of ``location`` and the transforms.

    Raises:
        ValueError: If ``location`` names ``activation``, ``coordinates`` is combined with ``location``,
            ``filter`` or a transform, the coordinates name different axes, or ``activation`` is set both
            on the locator and in the coordinates.
    """

    activation: ModelLayerLocation | None = None
    location: list[ModelAxis] = field(default_factory=list)
    filter: Filter = field(default_factory=Filter)
    group: Group = field(default_factory=Group)
    incremental: Incremental = field(default_factory=Incremental)
    combination: Combination = field(default_factory=Combination)
    coordinates: Coordinates | None = None

    def __post_init__(self):
        hints = typing.get_type_hints(type(self))
        for declared in dataclasses.fields(self):
            setattr(self, declared.name, from_dict(hints[declared.name], getattr(self, declared.name)))

        if ModelAxis.ACTIVATION in self.location:
            raise ValueError("'activation' is not a declarative location; set 'activation' on the config instead")
        if self.coordinates is None:
            return
        if (
            self.location
            or self.filter.inclusion_terms
            or self.filter.exclusion_terms
            or self.group
            or self.incremental
            or self.combination.location
        ):
            raise ValueError(
                "'coordinates' cannot be combined with 'location'/'filter'/'group'/'incremental'/'combination'"
            )
        keys = set(self.coordinates[0].keys()) if self.coordinates else set()
        for coordinate in self.coordinates:
            if len(coordinate) == 0:
                raise ValueError(f"Invalid coordinate: {coordinate!r}")
            if set(coordinate.keys()) != keys:
                raise ValueError(
                    f"All coordinates must share the same keys {sorted(getattr(key, 'value', key) for key in keys)}:"
                    f" {coordinate!r}"
                )
        if ModelAxis.ACTIVATION in keys and self.activation is not None:
            raise ValueError("'activation' cannot be set both on the config and inside the coordinates")

    @property
    def _transforms(self) -> list[IndexTransform]:
        return [self.group, self.incremental, self.combination]

    def parse(self, model: Model, n_positions: int | None) -> ModelLocationIndex:
        """Resolve the locator into an index for a prompt of ``n_positions`` tokens.

        When the prompt length varies, as when probing prompts of unequal length, positions stay
        symbolic: a non-negative one counts from the start and a negative one from each prompt's own
        end, which torch resolves at slice time. Other axes never depend on the prompt length.

        Args:
            model: The model the axes are sized by.
            n_positions: The prompt's length, or ``None`` when it varies.

        Returns:
            The index, one row per location.

        Raises:
            ValueError: If ``n_positions`` is ``None`` and the locator would enumerate every position, or
                a filter or a coordinate is not valid.
        """
        if self.coordinates is not None:
            index_df = self.coordinates.to_df(model, n_positions, self.activation)
        else:
            location_to_max_value = {
                el: MODEL_LOCATION_MAPPER[el](model, self.activation, n_positions)
                for el in {*self.location, *self.group, *self.incremental}
            }
            index_df = make_df_from_ranges(
                self.location, [self.filter.location_range(el, location_to_max_value[el]) for el in self.location]
            )
            index_df = self.filter.apply(index_df, location_to_max_value)

            for transform in self._transforms:
                index_df = transform.apply(index_df, location_to_max_value)

        if ModelAxis.ACTIVATION not in index_df.columns:
            index_df[ModelAxis.ACTIVATION] = self.activation

        return ModelLocationIndex(index_df)
