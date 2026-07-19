import itertools
import sys
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from jaxtyping import Int
from pandas._typing import Axes
from transformer_lens import HookedTransformer

from intex.util.collections import range_combinations


class ModelLocation(str, Enum):
    ACTIVATION = "activation"
    LAYER = "layer"
    HEAD = "head"
    NEURON = "neuron"
    POSITION = "position"


class ModelLayerLocation(str, Enum):
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
    df: pd.DataFrame = field(default_factory=pd.DataFrame)


MODEL_LOCATION_MAPPER: dict[ModelLocation, Callable] = {
    ModelLocation.LAYER: lambda model, _1, _2: model.cfg.n_layers,
    ModelLocation.HEAD: lambda model, layer_location, _: (
        model.cfg.n_heads
        if layer_location not in {ModelLayerLocation.ATTENTION_K, ModelLayerLocation.ATTENTION_V}
        else model.cfg.n_key_value_heads
    ),
    ModelLocation.NEURON: (
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
    ModelLocation.POSITION: lambda _1, _2, prompt_size: prompt_size,
}


def make_df_from_ranges(column_names: Axes, column_ranges: Sequence[int | Sequence[int]]) -> pd.DataFrame:
    return pd.DataFrame(
        data=itertools.product(
            *[axis_range if isinstance(axis_range, Sequence) else range(axis_range) for axis_range in column_ranges]
        ),
        columns=column_names,
    )


class IndexTransform(Protocol):
    """A config section that rewrites the index DataFrame in place in the parse pipeline.

    ``location`` is the mixin's (mutable) list of locations; only ``Combination`` extends it.
    """

    def apply(self, index_df: pd.DataFrame) -> pd.DataFrame: ...


@dataclass
class Filter:
    """Keep/drop index rows from a list of patterns.

    Each pattern maps ``location -> constraint`` (multiple keys are AND-ed, patterns are OR-ed),
    and ``{"not": <pattern>}`` drops the matching rows instead. Pattern keys are ``ModelLocation``
    members (``from_config`` converts the config file's plain strings). A constraint is:
      int                     -> the value
      {"range": [start, end]} -> ``start..end`` inclusive
      [<value or range>, ...] -> any of the listed constraints
    Negative numbers count from the end. The result is the union of the positive patterns
    (everything when there are none) minus the union of the ``not`` patterns.
    """

    inclusion_terms: list[dict[ModelLocation, int | dict | list]] = field(default_factory=list)
    exclusion_terms: list[dict[ModelLocation, int | dict | list]] = field(default_factory=list)

    @classmethod
    def from_config(cls, terms: list) -> "Filter":
        inclusion_terms: list[dict] = []
        exclusion_terms: list[dict] = []
        for term in terms:
            pattern = term["not"] if isinstance(term, dict) and set(term.keys()) == {"not"} else term
            if not isinstance(pattern, dict) or len(pattern) == 0:
                raise ValueError(f"Invalid filter term: {term!r}")
            pattern = {ModelLocation(location): constraint for location, constraint in pattern.items()}

            if isinstance(term, dict) and set(term.keys()) == {"not"}:
                exclusion_terms.append(pattern)
            else:
                inclusion_terms.append(pattern)

        return cls(inclusion_terms=inclusion_terms, exclusion_terms=exclusion_terms)

    def included(self, location: ModelLocation, max_value: int) -> set[int]:
        if not self.inclusion_terms or any(location not in term for term in self.inclusion_terms):
            return set(range(max_value))
        return set().union(*(self._resolve(term[location], location, max_value) for term in self.inclusion_terms))

    def excluded(self, location: ModelLocation, max_value: int) -> set[int]:
        return set().union(
            *(
                self._resolve(term[location], location, max_value)
                for term in self.exclusion_terms
                if set(term.keys()) == {location}
            )
        )

    def location_range(self, location: ModelLocation, max_value: int) -> list[int]:
        return sorted(self.included(location, max_value) - self.excluded(location, max_value))

    def apply(self, index_df: pd.DataFrame, location_to_max_value: dict) -> pd.DataFrame:
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
    def _resolve(cls, constraint: int | dict | list, location: ModelLocation, max_value: int) -> set[int]:
        if isinstance(constraint, list):
            return set().union(*(cls._resolve(el, location, max_value) for el in constraint))
        if isinstance(constraint, int):
            return {constraint if constraint >= 0 else max_value + constraint}
        if isinstance(constraint, dict) and set(constraint.keys()) == {"range"}:
            start, end = constraint["range"]
            start = start if start >= 0 else max_value + start
            end = end if end >= 0 else max_value + end
            return set(range(start, end + 1))
        raise ValueError(f"Invalid filter constraint for '{location}': {constraint!r}")


class Group(dict[ModelLocation, list[int] | str]):
    """Group values together (collapsing the resulting duplicate rows)."""

    def apply(self, index_df: pd.DataFrame) -> pd.DataFrame:
        for name, values in self.items():
            if values == "all":
                values = index_df[name].unique().tolist()
            values_to_replace = {v: values for v in values}
            index_df[name] = index_df[name].apply((lambda mapping: lambda el: mapping.get(el, el))(values_to_replace))
        index_df = index_df.loc[index_df.astype(str).drop_duplicates().index]

        return index_df


class Incremental(dict[ModelLocation, list[int]]):
    """Create new incremental rows (taking all rows and incrementing the specified locations)."""

    def apply(self, index_df: pd.DataFrame) -> pd.DataFrame:
        for name, values in self.items():
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
    """Create new combination rows (taking all rows and combining the specified location's values)."""

    location: list[ModelLocation] = field(default_factory=list)
    max_elements: int = sys.maxsize

    def apply(self, index_df: pd.DataFrame) -> pd.DataFrame:
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


class Coordinates(list[dict[ModelLocation, int | list[int] | ModelLayerLocation]]):
    """Explicit coordinates, each a ``location -> value`` mapping.

    ``from_config`` converts the config file's plain strings: coordinate keys become
    ``ModelLocation`` members, and ``activation`` values become ``ModelLayerLocation`` members.
    ``to_df`` resolves the coordinates into an index DataFrame, negative values counting from the
    end and the ``activation`` argument serving as the fallback for coordinates without their own.
    """

    @classmethod
    def from_config(cls, coordinates: list) -> "Coordinates":
        if not isinstance(coordinates, list):
            raise ValueError(f"Expected a list of coordinates, got {coordinates!r}")
        for coordinate in coordinates:
            if not isinstance(coordinate, dict) or len(coordinate) == 0:
                raise ValueError(f"Invalid coordinate: {coordinate!r}")

        return cls(
            {
                ModelLocation(key): (
                    ModelLayerLocation(value) if ModelLocation(key) is ModelLocation.ACTIVATION else value
                )
                for key, value in coordinate.items()
            }
            for coordinate in coordinates
        )

    def to_df(
        self,
        model: HookedTransformer,
        prompt: Int[torch.Tensor, "batch pos"],
        activation: ModelLayerLocation | None = None,
    ) -> pd.DataFrame:
        keys = sorted(list(self[0].keys()) if self else [], key=lambda key: key == ModelLocation.ACTIVATION)
        return pd.DataFrame(
            {
                key: pd.Series(
                    [self._resolve_entry(model, prompt, coordinate, key, activation) for coordinate in self],
                    dtype=object,
                )
                for key in keys
            }
        )

    @classmethod
    def _resolve_entry(
        cls,
        model: HookedTransformer,
        prompt: Int[torch.Tensor, "batch pos"],
        coordinate: dict,
        key: ModelLocation,
        activation: ModelLayerLocation | None,
    ) -> int | list[int] | ModelLayerLocation:
        if key == ModelLocation.ACTIVATION:
            return coordinate[key]
        activation = coordinate.get(ModelLocation.ACTIVATION, activation)
        max_value = MODEL_LOCATION_MAPPER[key](model, activation, prompt.shape[1])
        return cls._resolve_value(coordinate[key], key, max_value)

    @classmethod
    def _resolve_value(cls, value: int | list[int], location: ModelLocation, max_value: int) -> int | list[int]:
        if isinstance(value, list):
            return [cls._resolve_value(el, location, max_value) for el in value]
        if isinstance(value, int):
            return value if value >= 0 else max_value + value
        raise ValueError(f"Invalid coordinate value for '{location}': {value!r}")


@dataclass(kw_only=True)
class ModelLocatorConfig:
    """Locate model activations either declaratively or explicitly.

    Declaratively, ``location`` names the index axes and the full range of every axis is narrowed
    by ``filter`` and reshaped by the transforms. Explicitly, ``coordinates`` enumerates the exact
    points as ``location -> value`` mappings (every coordinate using the same locations, values as
    in ``Group`` rows: an int or a list of ints patched together); the two modes are mutually
    exclusive. A coordinate may also carry its own ``activation`` — then every coordinate must, and
    the config-level one must be unset. Negative values count from the end, as in ``Filter``.
    """

    activation: ModelLayerLocation | None = None
    location: list[ModelLocation] = field(default_factory=list)
    filter: Filter = field(default_factory=Filter)
    group: Group = field(default_factory=Group)
    incremental: Incremental = field(default_factory=Incremental)
    combination: Combination = field(default_factory=Combination)
    coordinates: Coordinates | None = None

    def __post_init__(self):
        if ModelLocation.ACTIVATION in self.location:
            raise ValueError("'activation' is not a declarative location; set 'activation' on the config instead")
        if self.coordinates is None:
            return
        if not isinstance(self.coordinates, Coordinates):
            self.coordinates = Coordinates.from_config(self.coordinates)
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
        if ModelLocation.ACTIVATION in keys and self.activation is not None:
            raise ValueError("'activation' cannot be set both on the config and inside the coordinates")

    @property
    def _transforms(self) -> list[IndexTransform]:
        return [self.group, self.incremental, self.combination]

    def parse(self, model: HookedTransformer, prompt: Int[torch.Tensor, "batch pos"]) -> ModelLocationIndex:
        if self.coordinates is not None:
            index_df = self.coordinates.to_df(model, prompt, self.activation)
        else:
            location_to_max_value = {
                el: MODEL_LOCATION_MAPPER[el](model, self.activation, prompt.shape[1]) for el in self.location
            }
            index_df = make_df_from_ranges(
                self.location, [self.filter.location_range(el, location_to_max_value[el]) for el in self.location]
            )
            index_df = self.filter.apply(index_df, location_to_max_value)

            for transform in self._transforms:
                index_df = transform.apply(index_df)

        if ModelLocation.ACTIVATION not in index_df.columns:
            index_df[ModelLocation.ACTIVATION] = self.activation

        return ModelLocationIndex(index_df)
