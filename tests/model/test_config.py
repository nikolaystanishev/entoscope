from types import SimpleNamespace

import numpy as np
import pytest
import torch

from intex.model.config import (
    ACTIVATION_NAME_COLUMN,
    ModelLayerLocation,
    ModelLocation,
    ModelLocatorConfigMixin,
)
from intex.operation.config import from_dict


def make_model(n_layers=4, n_heads=4, n_key_value_heads=2, d_model=8, d_head=4, d_mlp=16):
    """A stand-in for ``HookedTransformer`` exposing only the ``cfg`` fields the mapper reads."""
    return SimpleNamespace(
        cfg=SimpleNamespace(
            n_layers=n_layers,
            n_heads=n_heads,
            n_key_value_heads=n_key_value_heads,
            d_model=d_model,
            d_head=d_head,
            d_mlp=d_mlp,
        )
    )


def make_prompt(prompt_size):
    """A stand-in ``[batch, pos]`` token tensor; ``parse`` only reads ``prompt.shape[1]``."""
    return torch.zeros((1, prompt_size), dtype=torch.long)


def parse_config(config, *, layer_location=ModelLayerLocation.RESIDUAL_POST, prompt_size=4, **model_kwargs):
    return from_dict(ModelLocatorConfigMixin, {"activation_name": layer_location, **config}).parse(
        make_model(**model_kwargs), make_prompt(prompt_size)
    )


def _norm(value):
    """Canonicalise a cell so lists/tuples/numpy ints compare equal regardless of container type."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(_norm(v) for v in value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def result_rows(index):
    df = index.df
    rows = [
        tuple(
            sorted((ModelLocation(col).value, _norm(row[col])) for col in df.columns if col != ACTIVATION_NAME_COLUMN)
        )
        for _, row in df.iterrows()
    ]
    return sorted(rows, key=repr)


def expected_rows(rows):
    canonical = (tuple(sorted((loc, _norm(val)) for loc, val in row.items())) for row in rows)
    return sorted(canonical, key=repr)


def single(location, values):
    """Expected rows for a single-location index, one row per value."""
    return [{location: value} for value in values]


# --------------------------------------------------------------------------------------------------
# Location sizing — MODEL_LOCATION_MAPPER picks the right model dimension per location/layer_location
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "location, layer_location, model_kwargs, prompt_size, expected_values",
    [
        ("layer", ModelLayerLocation.RESIDUAL_POST, {"n_layers": 3}, 4, [0, 1, 2]),
        ("head", ModelLayerLocation.RESIDUAL_POST, {"n_heads": 4}, 4, [0, 1, 2, 3]),
        ("head", ModelLayerLocation.ATTENTION_K, {"n_heads": 4, "n_key_value_heads": 2}, 4, [0, 1]),
        ("head", ModelLayerLocation.ATTENTION_V, {"n_heads": 4, "n_key_value_heads": 2}, 4, [0, 1]),
        ("neuron", ModelLayerLocation.RESIDUAL_PRE, {"d_model": 2}, 4, [0, 1]),
        ("neuron", ModelLayerLocation.MLP_POST, {"d_mlp": 3}, 4, [0, 1, 2]),
        ("neuron", ModelLayerLocation.ATTENTION_Q, {"d_head": 2}, 4, [0, 1]),
        ("neuron", ModelLayerLocation.ATTENTION_PATTERN, {}, 4, []),  # pattern maps to 0 -> no neuron rows
        ("position", ModelLayerLocation.RESIDUAL_POST, {}, 3, [0, 1, 2]),
    ],
)
def test_location_sizing(location, layer_location, model_kwargs, prompt_size, expected_values):
    result = parse_config(
        {"location": [location]}, layer_location=layer_location, prompt_size=prompt_size, **model_kwargs
    )
    assert result_rows(result) == expected_rows(single(location, expected_values))


def test_empty_location_has_only_activation_column():
    # With no locations the index has no coordinate columns (the product over zero axes leaves a
    # single empty row); only the stamped activation column remains.
    result = parse_config({"location": []})
    assert list(result.df.columns) == [ACTIVATION_NAME_COLUMN]


def test_missing_location_key_matches_empty_list():
    # No ``location`` key at all behaves like an empty list.
    assert list(parse_config({}).df.columns) == [ACTIVATION_NAME_COLUMN]


def test_location_cartesian_product():
    result = parse_config({"location": ["layer", "head"]}, n_layers=2, n_heads=3)
    assert result_rows(result) == expected_rows(
        [{"layer": layer, "head": head} for layer in range(2) for head in range(3)]
    )


# --------------------------------------------------------------------------------------------------
# Activation column — the configured ``activation_name`` is stamped onto every row of the index
# --------------------------------------------------------------------------------------------------
def test_activation_name_column_is_stamped():
    result = parse_config({"location": ["layer"]}, layer_location=ModelLayerLocation.ATTENTION_OUT, n_layers=3)
    assert ACTIVATION_NAME_COLUMN in result.df.columns
    assert list(result.df[ACTIVATION_NAME_COLUMN]) == [ModelLayerLocation.ATTENTION_OUT.value] * 3


# --------------------------------------------------------------------------------------------------
# Filter — single location (covers int / range / list / negatives / inclusion / exclusion)
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "filter_cfg, n_layers, expected_layers",
    [
        # no filter -> the full range (docstring: "everything when there are none")
        ([], 4, [0, 1, 2, 3]),
        # inclusion constraints
        ([{"layer": 1}], 5, [1]),
        ([{"layer": {"range": [1, 3]}}], 5, [1, 2, 3]),
        ([{"layer": [0, 2, 4]}], 5, [0, 2, 4]),
        ([{"layer": [0, {"range": [2, 3]}]}], 5, [0, 2, 3]),
        # negatives count from the end
        ([{"layer": -1}], 4, [3]),
        ([{"layer": {"range": [-2, -1]}}], 4, [2, 3]),
        # exclusion only
        ([{"not": {"layer": 0}}], 3, [1, 2]),
        # inclusion AND exclusion together
        ([{"layer": {"range": [0, 3]}}, {"not": {"layer": 1}}], 5, [0, 2, 3]),
        # multiple inclusion terms are OR-ed
        ([{"layer": 0}, {"layer": 4}], 5, [0, 4]),
    ],
)
def test_filter_single_location(filter_cfg, n_layers, expected_layers):
    result = parse_config({"location": ["layer"], "filter": filter_cfg}, n_layers=n_layers)
    assert result_rows(result) == expected_rows(single("layer", expected_layers))


# --------------------------------------------------------------------------------------------------
# Filter — multiple locations (keys within a term AND, terms OR)
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "filter_cfg, expected",
    [
        # one term with two keys -> AND
        ([{"layer": 0, "head": 1}], [{"layer": 0, "head": 1}]),
        # two terms -> OR (layer == 0 for any head, OR head == 1 for any layer)
        (
            [{"layer": 0}, {"head": 1}],
            [
                {"layer": 0, "head": 0},
                {"layer": 0, "head": 1},
                {"layer": 0, "head": 2},
                {"layer": 1, "head": 1},
                {"layer": 2, "head": 1},
            ],
        ),
    ],
)
def test_filter_multi_location(filter_cfg, expected):
    result = parse_config({"location": ["layer", "head"], "filter": filter_cfg}, n_layers=3, n_heads=3)
    assert result_rows(result) == expected_rows(expected)


# --------------------------------------------------------------------------------------------------
# Group — collapse listed values into a shared list, dropping the resulting duplicate rows
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "config, model_kwargs, expected",
    [
        # explicit values 0 and 1 collapse into [0, 1]; 2 is left untouched
        ({"location": ["layer"], "group": {"layer": [0, 1]}}, {"n_layers": 3}, [{"layer": [0, 1]}, {"layer": 2}]),
        # "all" collapses every value into a single grouped row
        ({"location": ["layer"], "group": {"layer": "all"}}, {"n_layers": 3}, [{"layer": [0, 1, 2]}]),
        # grouping one axis collapses the duplicate rows it creates across the other axis
        (
            {"location": ["layer", "head"], "group": {"layer": [0, 1]}},
            {"n_layers": 2, "n_heads": 2},
            [{"layer": [0, 1], "head": 0}, {"layer": [0, 1], "head": 1}],
        ),
    ],
)
def test_group(config, model_kwargs, expected):
    result = parse_config(config, **model_kwargs)
    assert result_rows(result) == expected_rows(expected)


# --------------------------------------------------------------------------------------------------
# Incremental — append incrementing prefixes of the configured values to every row
# --------------------------------------------------------------------------------------------------
def test_incremental_single_location():
    result = parse_config({"location": ["layer"], "incremental": {"layer": [7, 8]}}, n_layers=2)
    assert result_rows(result) == expected_rows(
        [
            {"layer": [0]},
            {"layer": [0, 7]},
            {"layer": [0, 7, 8]},
            {"layer": [1]},
            {"layer": [1, 7]},
            {"layer": [1, 7, 8]},
        ]
    )


def test_incremental_keeps_other_locations():
    config = {"location": ["layer", "head"], "filter": [{"layer": 0}], "incremental": {"head": [9]}}
    result = parse_config(config, n_layers=2, n_heads=2)
    assert result_rows(result) == expected_rows(
        [
            {"layer": 0, "head": [0]},
            {"layer": 0, "head": [0, 9]},
            {"layer": 0, "head": [1]},
            {"layer": 0, "head": [1, 9]},
        ]
    )


def test_incremental_not_filtered():
    config = {
        "location": ["layer"],
        "filter": [{"layer": 0}],
        "incremental": {"neuron": [10, 11]},
    }
    result = parse_config(config, n_layers=2)
    assert result_rows(result) == expected_rows(
        [
            {"layer": 0, "neuron": [10]},
            {"layer": 0, "neuron": [10, 11]},
        ]
    )


# --------------------------------------------------------------------------------------------------
# Combination — expand a list-valued location into all of its value combinations
# (runs after ``group``, which is what turns the cells into lists in the first place)
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "combination_cfg, expected",
    [
        # max_elements caps the combination size at 2
        (
            {"location": ["layer"], "max_elements": 2},
            [
                {"layer": (0,)},
                {"layer": (1,)},
                {"layer": (2,)},
                {"layer": (0, 1)},
                {"layer": (0, 2)},
                {"layer": (1, 2)},
            ],
        ),
        # without max_elements every non-empty combination is produced (sizes 1..n)
        (
            {"location": ["layer"]},
            [
                {"layer": (0,)},
                {"layer": (1,)},
                {"layer": (2,)},
                {"layer": (0, 1)},
                {"layer": (0, 2)},
                {"layer": (1, 2)},
                {"layer": (0, 1, 2)},
            ],
        ),
    ],
)
def test_combination_after_group(combination_cfg, expected):
    config = {"location": ["layer"], "group": {"layer": "all"}, "combination": combination_cfg}
    result = parse_config(config, n_layers=3)
    assert result_rows(result) == expected_rows(expected)


# --------------------------------------------------------------------------------------------------
# Full pipeline — several sections together, exercising filter -> group -> combination ordering
# --------------------------------------------------------------------------------------------------
def test_full_config_filter_group_combination():
    config = {
        "location": ["layer"],
        "filter": [{"layer": {"range": [0, 2]}}],
        "group": {"layer": "all"},
        "combination": {"location": ["layer"], "max_elements": 2},
    }
    result = parse_config(config, n_layers=4)
    assert result_rows(result) == expected_rows(
        [{"layer": (0,)}, {"layer": (1,)}, {"layer": (2,)}, {"layer": (0, 1)}, {"layer": (0, 2)}, {"layer": (1, 2)}]
    )


def test_full_config_filter_group_multi_location():
    config = {"location": ["layer", "head"], "filter": [{"layer": {"range": [0, 1]}}], "group": {"head": "all"}}
    result = parse_config(config, n_layers=3, n_heads=2)
    assert result_rows(result) == expected_rows([{"layer": 0, "head": [0, 1]}, {"layer": 1, "head": [0, 1]}])


# --------------------------------------------------------------------------------------------------
# Error cases raised while *deserializing* the mixin (Filter.from_config / ModelLocation enum / strict keys)
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "config",
    [
        {"filter": [{}]},  # empty pattern
        {"filter": [{"not": {}}]},  # empty negated pattern
        {"filter": ["foo"]},  # term is not a dict
        {"filter": [5]},  # term is not a dict
        {"location": ["bogus"]},  # not a ModelLocation value
        {"fitler": [{"layer": 0}]},  # unknown key (typo) is rejected, not ignored
    ],
)
def test_construction_raises_value_error(config):
    with pytest.raises(ValueError):
        from_dict(ModelLocatorConfigMixin, {"activation_name": ModelLayerLocation.RESIDUAL_POST, **config})


# --------------------------------------------------------------------------------------------------
# Error cases that parse cleanly at construction but raise during parse()
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "config",
    [
        {"location": ["layer"], "filter": [{"layer": "x"}]},  # invalid constraint type
        {"location": ["layer"], "filter": [{"head": 0}]},  # references a location not in the index
    ],
)
def test_parse_raises_value_error(config):
    mixin = from_dict(
        ModelLocatorConfigMixin, {"activation_name": ModelLayerLocation.RESIDUAL_POST, **config}
    )  # deserialization succeeds
    with pytest.raises(ValueError):
        mixin.parse(make_model(), make_prompt(4))
