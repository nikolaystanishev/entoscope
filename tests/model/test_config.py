"""``ModelLocatorConfig`` — the shape a config file provides, and the index it resolves to.

Two steps are under test and nothing else: ``from_dict``, which turns the raw mapping into the config,
and ``parse(model, n_positions)``, which resolves it into an index DataFrame. The model is real because
every axis is sized from its ``cfg`` — the layer count, the head count, ``d_model`` / ``d_mlp`` /
``d_head`` — but no forward pass happens here.
"""

import numpy as np
import pandas as pd
import pytest

from entoscope.model.config import (
    Combination,
    Coordinates,
    Filter,
    Group,
    Incremental,
    ModelLayerLocation,
    ModelLocation,
    ModelLocatorConfig,
)
from entoscope.operation.config import from_dict

PROMPT_SIZE = 4


def parse(model, n_positions: int | None = PROMPT_SIZE, **source) -> pd.DataFrame:
    """The index a code-built config resolves to on this model."""
    return ModelLocatorConfig(**source).parse(model, n_positions).df


def deserialize(raw: dict, **overrides) -> ModelLocatorConfig:
    """The config a raw mapping (a YAML file's shape) deserializes into."""
    return from_dict(ModelLocatorConfig, {**raw, **overrides})


def _cell(value):
    """Canonicalise a cell so lists, tuples and numpy ints compare equal regardless of container."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(_cell(element) for element in value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def coordinates_of(index_df: pd.DataFrame, *locations: ModelLocation) -> list[tuple]:
    return [tuple(_cell(row[location]) for location in locations) for _, row in index_df.iterrows()]


def values_of(index_df: pd.DataFrame, location: ModelLocation) -> list:
    return [coordinate[0] for coordinate in coordinates_of(index_df, location)]


# --------------------------------------------------------------------------------------------------
# Location axes — every location enumerates the model dimension it names
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("model_fixture", "activation", "location", "axis_size"),
    [
        ("small_model", ModelLayerLocation.RESIDUAL_POST, ModelLocation.LAYER, lambda cfg: cfg.n_layers),
        ("small_model", ModelLayerLocation.ATTENTION_Z, ModelLocation.HEAD, lambda cfg: cfg.n_heads),
        # Key/value heads are their own axis: grouped-query attention has fewer of them than query heads,
        # while multi-head attention leaves n_key_value_heads unset and the axis is the query heads.
        ("gqa_model", ModelLayerLocation.ATTENTION_K, ModelLocation.HEAD, lambda cfg: cfg.n_key_value_heads),
        ("gqa_model", ModelLayerLocation.ATTENTION_V, ModelLocation.HEAD, lambda cfg: cfg.n_key_value_heads),
        ("small_model", ModelLayerLocation.ATTENTION_K, ModelLocation.HEAD, lambda cfg: cfg.n_heads),
        ("small_model", ModelLayerLocation.RESIDUAL_PRE, ModelLocation.NEURON, lambda cfg: cfg.d_model),
        ("small_model", ModelLayerLocation.MLP_POST, ModelLocation.NEURON, lambda cfg: cfg.d_mlp),
        ("small_model", ModelLayerLocation.ATTENTION_Q, ModelLocation.NEURON, lambda cfg: cfg.d_head),
        # The pattern has no neuron axis at all, so the location contributes no rows.
        ("small_model", ModelLayerLocation.ATTENTION_PATTERN, ModelLocation.NEURON, lambda cfg: 0),
        # The position axis is the only one sized by the prompt rather than the model.
        ("small_model", ModelLayerLocation.RESIDUAL_POST, ModelLocation.POSITION, lambda cfg: PROMPT_SIZE),
    ],
)
def test_a_location_enumerates_the_model_dimension_it_names(
    request, backend, model_fixture, activation, location, axis_size
):
    model = request.getfixturevalue(model_fixture)

    index_df = parse(model, activation=activation, location=[location])

    assert values_of(index_df, location) == list(range(axis_size(model.cfg)))


def test_locations_form_a_cartesian_product(small_model):
    index_df = parse(
        small_model,
        activation=ModelLayerLocation.MLP_POST,
        location=[ModelLocation.LAYER, ModelLocation.NEURON],
    )

    assert coordinates_of(index_df, ModelLocation.LAYER, ModelLocation.NEURON) == [
        (layer, neuron) for layer in range(small_model.cfg.n_layers) for neuron in range(small_model.cfg.d_mlp)
    ]


def test_the_configured_activation_is_stamped_on_every_row(small_model):
    index_df = parse(small_model, activation=ModelLayerLocation.ATTENTION_OUT, location=[ModelLocation.LAYER])

    assert ModelLocation.ACTIVATION in index_df.columns
    assert (
        values_of(index_df, ModelLocation.ACTIVATION) == [ModelLayerLocation.ATTENTION_OUT] * small_model.cfg.n_layers
    )


def test_without_a_location_only_the_activation_is_left(small_model):
    # The product over zero axes leaves a single empty row, so nothing but the stamped activation remains.
    from_empty = parse(small_model, activation=ModelLayerLocation.RESIDUAL_POST, location=[])
    from_missing = parse(small_model, activation=ModelLayerLocation.RESIDUAL_POST)

    assert list(from_empty.columns) == list(from_missing.columns) == [ModelLocation.ACTIVATION]
    assert len(from_empty) == len(from_missing) == 1


# --------------------------------------------------------------------------------------------------
# Filter — inclusion terms are OR-ed, keys within a term AND-ed, exclusions subtracted
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("filter_config", "expected_layers"),
    [
        ([], list(range(12))),  # no filter keeps the full range
        ([{"layer": 1}], [1]),
        ([{"layer": {"range": [1, 3]}}], [1, 2, 3]),
        ([{"layer": [0, 2, 4]}], [0, 2, 4]),
        ([{"layer": [0, {"range": [2, 3]}]}], [0, 2, 3]),
        ([{"layer": -1}], [11]),  # negatives count from the end
        ([{"layer": {"range": [-2, -1]}}], [10, 11]),
        ([{"not": {"layer": 0}}], list(range(1, 12))),  # exclusion only
        ([{"layer": {"range": [0, 3]}}, {"not": {"layer": 1}}], [0, 2, 3]),  # inclusion minus exclusion
        ([{"layer": 0}, {"layer": 4}], [0, 4]),  # separate terms are OR-ed
    ],
)
def test_a_filter_narrows_a_single_axis(model, filter_config, expected_layers):
    # GPT-2's 12 layers give the ranges and negative anchors above room to differ from each other.
    assert model.cfg.n_layers == 12

    index_df = parse(
        model,
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelLocation.LAYER],
        filter=Filter.from_config(filter_config),
    )

    assert values_of(index_df, ModelLocation.LAYER) == expected_layers


def test_the_keys_within_a_filter_term_are_and_ed(small_model):
    index_df = parse(
        small_model,
        activation=ModelLayerLocation.ATTENTION_Z,
        location=[ModelLocation.LAYER, ModelLocation.HEAD],
        filter=Filter.from_config([{"layer": 0, "head": 1}]),
    )

    assert coordinates_of(index_df, ModelLocation.LAYER, ModelLocation.HEAD) == [(0, 1)]


def test_filter_terms_are_or_ed_across_locations(small_model):
    # layer 0 for any head, OR head 1 for any layer.
    index_df = parse(
        small_model,
        activation=ModelLayerLocation.ATTENTION_Z,
        location=[ModelLocation.LAYER, ModelLocation.HEAD],
        filter=Filter.from_config([{"layer": 0}, {"head": 1}]),
    )

    assert set(coordinates_of(index_df, ModelLocation.LAYER, ModelLocation.HEAD)) == {
        *((0, head) for head in range(small_model.cfg.n_heads)),
        *((layer, 1) for layer in range(small_model.cfg.n_layers)),
    }


# --------------------------------------------------------------------------------------------------
# Transforms — group collapses values into one cell, incremental grows prefixes, combination expands
# a grouped cell again. They run in that order, after the filter.
# --------------------------------------------------------------------------------------------------
def _layers(model, **source) -> list:
    return values_of(parse(model, activation=ModelLayerLocation.RESIDUAL_POST, **source), ModelLocation.LAYER)


def test_group_collapses_the_listed_values_into_one_row(small_model):
    # 0 and 1 collapse into one row addressing both; 2 is left untouched.
    assert _layers(small_model, location=[ModelLocation.LAYER], group=Group({ModelLocation.LAYER: [0, 1]})) == [
        (0, 1),
        2,
    ]


def test_group_all_collapses_the_whole_axis(small_model):
    layers = _layers(small_model, location=[ModelLocation.LAYER], group=Group({ModelLocation.LAYER: "all"}))

    assert layers == [tuple(range(small_model.cfg.n_layers))]


def test_group_drops_the_duplicate_rows_it_creates(small_model):
    # Grouping one axis leaves one row per value of the other, not one per pair.
    index_df = parse(
        small_model,
        activation=ModelLayerLocation.ATTENTION_Z,
        location=[ModelLocation.LAYER, ModelLocation.HEAD],
        group=Group({ModelLocation.LAYER: "all"}),
    )

    assert coordinates_of(index_df, ModelLocation.LAYER, ModelLocation.HEAD) == [
        (tuple(range(small_model.cfg.n_layers)), head) for head in range(small_model.cfg.n_heads)
    ]


def test_incremental_appends_growing_prefixes_to_every_row(small_model):
    layers = _layers(
        small_model,
        location=[ModelLocation.LAYER],
        filter=Filter.from_config([{"layer": [0, 1]}]),
        incremental=Incremental({ModelLocation.LAYER: [7, 8]}),
    )

    assert layers == [(0,), (0, 7), (0, 7, 8), (1,), (1, 7), (1, 7, 8)]


def test_incremental_can_add_a_location_the_index_does_not_have(small_model):
    # The neuron axis is not in the index, so only the incremented rows exist.
    index_df = parse(
        small_model,
        activation=ModelLayerLocation.MLP_POST,
        location=[ModelLocation.LAYER],
        filter=Filter.from_config([{"layer": 0}]),
        incremental=Incremental({ModelLocation.NEURON: [4, 5]}),
    )

    assert coordinates_of(index_df, ModelLocation.LAYER, ModelLocation.NEURON) == [(0, (4,)), (0, (4, 5))]


def test_incremental_keeps_the_other_locations(small_model):
    index_df = parse(
        small_model,
        activation=ModelLayerLocation.ATTENTION_Z,
        location=[ModelLocation.LAYER, ModelLocation.HEAD],
        filter=Filter.from_config([{"layer": 0, "head": [0, 1]}]),
        incremental=Incremental({ModelLocation.HEAD: [9]}),
    )

    assert coordinates_of(index_df, ModelLocation.LAYER, ModelLocation.HEAD) == [
        (0, (0,)),
        (0, (0, 9)),
        (0, (1,)),
        (0, (1, 9)),
    ]


@pytest.mark.parametrize(
    ("max_elements", "expected"),
    [
        # max_elements caps the size of a combination...
        (2, [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]),
        # ...and without it every non-empty combination is produced.
        (None, [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]),
    ],
)
def test_combination_expands_a_grouped_axis(small_model, max_elements, expected):
    assert small_model.cfg.n_layers == 3

    layers = _layers(
        small_model,
        location=[ModelLocation.LAYER],
        group=Group({ModelLocation.LAYER: "all"}),
        combination=Combination(
            location=[ModelLocation.LAYER], **({} if max_elements is None else {"max_elements": max_elements})
        ),
    )

    assert layers == expected


def test_filter_group_and_combination_run_in_that_order(model):
    # The filter narrows the axis to 0..2 first, so the combinations never mention any later layer.
    layers = _layers(
        model,
        location=[ModelLocation.LAYER],
        filter=Filter.from_config([{"layer": {"range": [0, 2]}}]),
        group=Group({ModelLocation.LAYER: "all"}),
        combination=Combination(location=[ModelLocation.LAYER], max_elements=2),
    )

    assert layers == [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]


# --------------------------------------------------------------------------------------------------
# Explicit coordinates — the exact points, in the order they are listed
# --------------------------------------------------------------------------------------------------
def test_coordinates_are_indexed_in_order(model):
    index_df = parse(
        model,
        activation=ModelLayerLocation.RESIDUAL_POST,
        coordinates=[
            {ModelLocation.LAYER: 2, ModelLocation.POSITION: 1},
            {ModelLocation.LAYER: 0, ModelLocation.POSITION: 3},
        ],
    )

    assert coordinates_of(index_df, ModelLocation.LAYER, ModelLocation.POSITION) == [(2, 1), (0, 3)]
    assert values_of(index_df, ModelLocation.ACTIVATION) == [ModelLayerLocation.RESIDUAL_POST] * 2


def test_coordinates_resolve_negative_values(model):
    index_df = parse(
        model,
        activation=ModelLayerLocation.MLP_POST,
        coordinates=[{ModelLocation.LAYER: -1, ModelLocation.NEURON: -2}],
    )

    assert coordinates_of(index_df, ModelLocation.LAYER, ModelLocation.NEURON) == [
        (model.cfg.n_layers - 1, model.cfg.d_mlp - 2)
    ]


def test_a_coordinate_list_addresses_several_values_in_one_row(model):
    # A list value addresses those elements together in one row; negatives resolve inside lists too.
    index_df = parse(
        model,
        activation=ModelLayerLocation.RESIDUAL_POST,
        coordinates=[{ModelLocation.LAYER: 1, ModelLocation.POSITION: [0, -1]}],
    )

    assert coordinates_of(index_df, ModelLocation.POSITION) == [((0, PROMPT_SIZE - 1),)]


def test_coordinates_carry_their_own_activation(model):
    config = deserialize(
        {"coordinates": [{"activation": "resid.post", "layer": 0}, {"activation": "mlp.out", "layer": -1}]}
    )

    # The deserializer turns the values into members, and parse orders the activation column last.
    assert all(
        isinstance(coordinate[ModelLocation.ACTIVATION], ModelLayerLocation) for coordinate in config.coordinates
    )
    index_df = config.parse(model, PROMPT_SIZE).df
    assert list(index_df.columns) == [ModelLocation.LAYER, ModelLocation.ACTIVATION]
    assert values_of(index_df, ModelLocation.ACTIVATION) == [
        ModelLayerLocation.RESIDUAL_POST,
        ModelLayerLocation.MLP_OUT,
    ]
    assert values_of(index_df, ModelLocation.LAYER) == [0, model.cfg.n_layers - 1]


def test_a_negative_coordinate_resolves_against_its_own_activations_dimension(model):
    # The same {neuron: -1} is d_head-1 for a head activation and d_mlp-1 for an MLP one.
    config = deserialize(
        {"coordinates": [{"activation": "attn.q", "neuron": -1}, {"activation": "mlp.post", "neuron": -1}]}
    )

    assert values_of(config.parse(model, PROMPT_SIZE).df, ModelLocation.NEURON) == [
        model.cfg.d_head - 1,
        model.cfg.d_mlp - 1,
    ]


def test_an_empty_coordinate_list_yields_no_rows(model):
    index_df = parse(model, activation=ModelLayerLocation.RESIDUAL_POST, coordinates=[])

    assert len(index_df) == 0
    assert ModelLocation.ACTIVATION in index_df.columns


# --------------------------------------------------------------------------------------------------
# Deserialization — the config file's strings become ModelLocation / ModelLayerLocation members
# --------------------------------------------------------------------------------------------------
def test_every_section_is_keyed_by_model_location(model):
    config = deserialize(
        {
            "activation": "resid.post",
            "location": ["layer"],
            "filter": [{"layer": 0}, {"not": {"head": 1}}],
            "group": {"layer": "all"},
            "incremental": {"head": [1]},
            "combination": {"location": ["layer"]},
        }
    )

    assert config.activation is ModelLayerLocation.RESIDUAL_POST
    assert config.location == [ModelLocation.LAYER]
    terms = config.filter.inclusion_terms + config.filter.exclusion_terms
    assert all(isinstance(location, ModelLocation) for term in terms for location in term)
    assert isinstance(config.group, Group)
    assert isinstance(config.incremental, Incremental)
    assert isinstance(config.combination, Combination)
    assert all(isinstance(location, ModelLocation) for location in [*config.group, *config.incremental])
    assert all(isinstance(location, ModelLocation) for location in config.combination.location)


def test_coordinate_keys_are_model_location(model):
    config = deserialize({"coordinates": [{"activation": "resid.post", "layer": 0, "position": -1}]})

    assert isinstance(config.coordinates, Coordinates)
    assert all(isinstance(location, ModelLocation) for coordinate in config.coordinates for location in coordinate)


def test_a_raw_config_parses_like_its_code_built_twin(model):
    raw = deserialize(
        {
            "activation": "resid.post",
            "location": ["layer"],
            "filter": [{"layer": {"range": [0, 2]}}],
            "group": {"layer": "all"},
            "combination": {"location": ["layer"], "max_elements": 2},
        }
    )
    in_code = ModelLocatorConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelLocation.LAYER],
        filter=Filter(inclusion_terms=[{ModelLocation.LAYER: {"range": [0, 2]}}]),
        group=Group({ModelLocation.LAYER: "all"}),
        combination=Combination(location=[ModelLocation.LAYER], max_elements=2),
    )

    assert raw == in_code
    assert values_of(raw.parse(model, PROMPT_SIZE).df, ModelLocation.LAYER) == [
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 2),
    ]


# --------------------------------------------------------------------------------------------------
# Symbolic positions — parse(model, None) keeps a position unresolved when the prompt length varies
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("filter_config", "expected_positions"),
    [
        ([{"position": 2}], [2]),  # non-negative stays absolute, counted from the start
        ([{"position": -1}], [-1]),  # negative stays symbolic, resolved per prompt at slice time
        ([{"position": 0}, {"position": -1}], [-1, 0]),  # mixed anchors are two distinct points
        ([{"position": [1, -2]}], [-2, 1]),  # a list keeps each element's sign
        ([{"position": {"range": [0, 2]}}], [0, 1, 2]),  # a non-negative range expands from the start
        ([{"position": {"range": [-3, -1]}}], [-3, -2, -1]),  # a negative range expands in end space
    ],
)
def test_a_position_stays_symbolic_when_the_prompt_length_varies(small_model, filter_config, expected_positions):
    index_df = parse(
        small_model,
        n_positions=None,
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelLocation.POSITION],
        filter=Filter.from_config(filter_config),
    )

    assert values_of(index_df, ModelLocation.POSITION) == expected_positions


def test_a_symbolic_position_keeps_the_other_locations(small_model):
    index_df = parse(
        small_model,
        n_positions=None,
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelLocation.LAYER, ModelLocation.POSITION],
        filter=Filter.from_config([{"position": -1}]),
    )

    assert coordinates_of(index_df, ModelLocation.LAYER, ModelLocation.POSITION) == [
        (layer, -1) for layer in range(small_model.cfg.n_layers)
    ]


def test_a_known_prompt_length_resolves_a_negative_position(small_model):
    # The contrast: with a length to count against, -1 collapses to its absolute index.
    index_df = parse(
        small_model,
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelLocation.POSITION],
        filter=Filter.from_config([{"position": -1}]),
    )

    assert values_of(index_df, ModelLocation.POSITION) == [PROMPT_SIZE - 1]


def test_symbolic_coordinates_keep_their_signs(small_model):
    index_df = parse(
        small_model,
        n_positions=None,
        activation=ModelLayerLocation.RESIDUAL_POST,
        coordinates=[
            {ModelLocation.LAYER: 1, ModelLocation.POSITION: -2},
            {ModelLocation.LAYER: 0, ModelLocation.POSITION: [0, -1]},
        ],
    )

    assert coordinates_of(index_df, ModelLocation.LAYER, ModelLocation.POSITION) == [(1, -2), (0, (0, -1))]


def test_a_transform_treats_a_symbolic_position_like_any_other_value(small_model):
    # Group collapses negative positions just as it collapses non-negative ones.
    index_df = parse(
        small_model,
        n_positions=None,
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelLocation.POSITION],
        filter=Filter.from_config([{"position": {"range": [-3, -1]}}]),
        group=Group({ModelLocation.POSITION: [-2, -1]}),
    )

    assert values_of(index_df, ModelLocation.POSITION) == [-3, (-2, -1)]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        # an unfiltered position axis would enumerate every position
        ({"location": [ModelLocation.POSITION]}, "prompt length varies"),
        # an exclusion still needs the full range to subtract from
        (
            {
                "location": [ModelLocation.POSITION],
                "filter": Filter.from_config([{"not": {"position": 0}}]),
            },
            "prompt length varies",
        ),
        # a mixed-sign range has no length-independent meaning
        (
            {
                "location": [ModelLocation.POSITION],
                "filter": Filter.from_config([{"position": {"range": [-1, 1]}}]),
            },
            "positions on the same side",
        ),
    ],
)
def test_a_position_grid_that_depends_on_the_prompt_length_raises(small_model, source, message):
    with pytest.raises(ValueError, match=message):
        parse(small_model, n_positions=None, activation=ModelLayerLocation.RESIDUAL_POST, **source)


# --------------------------------------------------------------------------------------------------
# Rejected configurations
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "source",
    [
        {"filter": [{}]},  # empty pattern
        {"filter": [{"not": {}}]},  # empty negated pattern
        {"filter": ["foo"]},  # term is not a mapping
        {"filter": [5]},
        {"location": ["bogus"]},  # not a ModelLocation value
        {"fitler": [{"layer": 0}]},  # a typo is rejected, not ignored
        {"filter": [{"bogus": 0}]},
        {"group": {"bogus": "all"}},
        {"group": "all"},  # a section that keys values by location needs a mapping
        {"incremental": [1, 2]},
        {"location": ["activation"]},  # activation is not a declarative location
        {"coordinates": [{"bogus": 0}]},
        {"coordinates": [{"layer": 0}], "location": ["layer"]},  # the two modes are exclusive
        {"coordinates": [{"layer": 0}], "filter": [{"layer": 0}]},
        {"coordinates": [{"layer": 0}, {"head": 0}]},  # coordinates do not share their locations
        {"coordinates": [{}]},  # empty coordinate
        {"coordinates": "layer"},  # coordinates are a list of mappings, not a scalar
        {"coordinates": [{"activation": "bogus", "layer": 0}]},  # invalid activation value
        {"activation": "resid.post", "coordinates": [{"activation": "resid.post", "layer": 0}]},  # activation twice
        {"coordinates": [{"activation": "resid.post", "layer": 0}, {"layer": 1}]},  # only some carry one
    ],
)
def test_a_malformed_source_is_rejected_when_it_is_deserialized(source):
    # Coordinates carry their own activation, so only the declarative shapes get the config-level one.
    default = {} if "coordinates" in source else {"activation": "resid.post"}

    with pytest.raises(ValueError):
        deserialize({**default, **source})


def test_a_code_built_coordinate_is_checked_too(model):
    # Built as ``Coordinates`` the deserializer's own checks are skipped, so the config repeats them.
    with pytest.raises(ValueError, match="Invalid coordinate"):
        ModelLocatorConfig(activation=ModelLayerLocation.RESIDUAL_POST, coordinates=Coordinates([{}]))


@pytest.mark.parametrize(
    "source",
    [
        {"location": ["layer"], "filter": [{"layer": "x"}]},  # invalid constraint type
        {"location": ["layer"], "filter": [{"head": 0}]},  # references a location the index does not have
        {"coordinates": [{"layer": "x"}]},  # invalid coordinate value type
    ],
)
def test_a_source_the_model_cannot_resolve_raises_when_it_is_parsed(small_model, source):
    config = deserialize({"activation": "resid.post", **source})  # deserialization succeeds

    with pytest.raises(ValueError):
        config.parse(small_model, PROMPT_SIZE)
