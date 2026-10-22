"""The locator a config file names, and the index a run over it iterates - as its dry run reports it.

Every locator here is a patch element read from a YAML file, and every index the one ``plan()``
resolves for it: what ``--dry-run`` prints and ``index.csv`` holds. Planning sizes every axis from the
model's ``cfg`` and the prompts' token counts, but runs no forward pass. A prompt length that varies
is a probing run's, so the symbolic positions are read off a probe's plan.
"""

import re

import numpy as np
import pandas as pd
import pytest

from entoscope.model.config import (
    Combination,
    Coordinates,
    Filter,
    Group,
    Incremental,
    ModelAxis,
    ModelLayerLocation,
)
from entoscope.patching.config import (
    MetricComparison,
    PatchingConfig,
    PatchingElementConfig,
    PatchingMethodConfig,
    PatchingMetric,
    PatchingMetricConfig,
    PatchingType,
    SinglePatchingConfig,
)
from entoscope.patching.input import PatchingInput
from entoscope.patching.patcher import Patcher
from entoscope.probe.config import ProbeConfig
from entoscope.probe.input import ProbeInput
from entoscope.probe.prober import Prober

from ..conftest import forbid_model_runs

SOURCE_PROMPT = "The capital of France is"
TARGET_PROMPT = "The capital of Spain is"
# Two prompts of different lengths, so a probe over them cannot resolve a position against either.
LONG_PROMPT, SHORT_PROMPT = "The movie was absolutely wonderful", "I loved it"

PATCHING = {
    "metric": {"probability": {"type": "probability", "tokens": ["correct"], "comparison": "absolute"}},
    "method": "exact",
}


def n_positions(model) -> int:
    return model.to_tokens(SOURCE_PROMPT).shape[1]


def _source_columns(index_df: pd.DataFrame) -> pd.DataFrame:
    source = [column for column in index_df.columns if column.startswith("source_")]
    return index_df[source].rename(columns=lambda column: column.removeprefix("source_"))


@pytest.fixture
def planned(config_from_yaml, monkeypatch):
    """The index a patching run over a locator would iterate, read off its plan.

    A mapping is the locator as a config file writes it; a ``PatchingElementConfig`` is one built in code.
    """

    def plan(model, locator: dict | PatchingElementConfig) -> pd.DataFrame:
        if isinstance(locator, dict):
            element = {**locator, "type": "replace"}
            config = config_from_yaml(
                PatchingConfig, "locator", {**PATCHING, "patch": [{"source": element, "target": element}]}
            )
        else:
            config = PatchingConfig(
                metric={
                    "probability": PatchingMetricConfig(
                        type=PatchingMetric.PROBABILITY, tokens=["correct"], comparison=MetricComparison.ABSOLUTE
                    )
                },
                method=PatchingMethodConfig(type="exact"),
                patch=[SinglePatchingConfig(source=locator, target=locator)],
            )
        forbid_model_runs(monkeypatch, model)
        patching_input = PatchingInput(config, SOURCE_PROMPT, TARGET_PROMPT, tokens={"correct": " Paris"})
        index = Patcher(model).plan(patching_input, verbose=False).index[0]
        return _source_columns(index)

    return plan


@pytest.fixture
def planned_symbolic(config_from_yaml, monkeypatch):
    """The index a probe over prompts of different lengths would iterate, read off its plan."""

    def plan(model, locator: dict) -> pd.DataFrame:
        config = config_from_yaml(
            ProbeConfig, "locator", {"method": "linear", "pool": {}, "metric": ["accuracy"], "source": locator}
        )
        forbid_model_runs(monkeypatch, model)
        index = Prober(model).plan(ProbeInput(config, [LONG_PROMPT], [SHORT_PROMPT]), verbose=False).index[0]
        return _source_columns(index)

    return plan


def _cell(value):
    """Canonicalise a cell so lists, tuples and numpy ints compare equal regardless of container."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(_cell(element) for element in value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def coordinates_of(index_df: pd.DataFrame, *locations: ModelAxis) -> list[tuple]:
    return [tuple(_cell(row[location]) for location in locations) for _, row in index_df.iterrows()]


def values_of(index_df: pd.DataFrame, location: ModelAxis) -> list:
    return [coordinate[0] for coordinate in coordinates_of(index_df, location)]


# --------------------------------------------------------------------------------------------------
# Location axes - every location enumerates the model dimension it names
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("model_fixture", "activation", "location", "axis_size"),
    [
        ("small_model", "resid.post", "layer", lambda model: model.cfg.n_layers),
        ("small_model", "attn.z", "head", lambda model: model.cfg.n_heads),
        # Key/value heads are their own axis: grouped-query attention has fewer of them than query heads,
        # while multi-head attention leaves n_key_value_heads unset and the axis is the query heads.
        ("gqa_model", "attn.k", "head", lambda model: model.cfg.n_key_value_heads),
        ("gqa_model", "attn.v", "head", lambda model: model.cfg.n_key_value_heads),
        ("small_model", "attn.k", "head", lambda model: model.cfg.n_heads),
        ("small_model", "resid.pre", "neuron", lambda model: model.cfg.d_model),
        ("small_model", "mlp.post", "neuron", lambda model: model.cfg.d_mlp),
        ("small_model", "attn.q", "neuron", lambda model: model.cfg.d_head),
        ("small_model", "attn.pattern", "head", lambda model: model.cfg.n_heads),
        # The position axes are the only ones sized by the prompt rather than the model.
        ("small_model", "resid.post", "position", n_positions),
        ("small_model", "attn.pattern", "dest_pos", n_positions),
        ("small_model", "attn.pattern", "src_pos", n_positions),
    ],
)
def test_a_location_enumerates_the_model_dimension_it_names(
    request, backend, planned, model_fixture, activation, location, axis_size
):
    model = request.getfixturevalue(model_fixture)

    index_df = planned(model, {"activation": activation, "location": [location]})

    assert values_of(index_df, ModelAxis(location)) == list(range(axis_size(model)))


SUBLAYER_HOOKS = ["attn.in", "attn.out_pre_norm", "mlp.in", "mlp.out_pre_norm"]


@pytest.mark.parametrize("activation", SUBLAYER_HOOKS)
def test_a_sublayers_own_hook_is_located_along_the_residual_stream(backend, small_model, planned, activation):
    if backend == "hooked":
        pytest.skip("a HookedTransformer has no hooks on its sublayers themselves")

    index_df = planned(small_model, {"activation": activation, "location": ["layer", "position", "neuron"]})

    assert coordinates_of(index_df, ModelAxis.LAYER, ModelAxis.POSITION, ModelAxis.NEURON) == [
        (layer, position, neuron)
        for layer in range(small_model.cfg.n_layers)
        for position in range(n_positions(small_model))
        for neuron in range(small_model.cfg.d_model)
    ]


@pytest.mark.parametrize("activation", SUBLAYER_HOOKS)
def test_a_hooked_transformer_refuses_a_sublayers_own_hook(backend, small_model, planned, activation):
    if backend != "hooked":
        pytest.skip("only a HookedTransformer lacks hooks on its sublayers themselves")

    # Every coordinate's activation is checked, not only the first's.
    locator = {"coordinates": [{"activation": "resid.post", "layer": 0}, {"activation": activation, "layer": 0}]}
    with pytest.raises(ValueError, match=re.escape(f"'{activation}' is a hook of a TransformerBridge's sublayer")):
        planned(small_model, locator)


def test_locations_form_a_cartesian_product(small_model, planned):
    index_df = planned(small_model, {"activation": "mlp.post", "location": ["layer", "neuron"]})

    assert coordinates_of(index_df, ModelAxis.LAYER, ModelAxis.NEURON) == [
        (layer, neuron) for layer in range(small_model.cfg.n_layers) for neuron in range(small_model.cfg.d_mlp)
    ]


def test_a_pattern_is_located_entry_by_entry(small_model, planned):
    # dest_pos is the query position attending and src_pos the key position attended to.
    index_df = planned(
        small_model,
        {
            "activation": "attn.pattern",
            "location": ["layer", "head", "dest_pos", "src_pos"],
            "filter": [{"layer": 0, "head": 1, "dest_pos": -1}],
        },
    )

    last = n_positions(small_model) - 1
    assert coordinates_of(
        index_df, ModelAxis.LAYER, ModelAxis.HEAD, ModelAxis.DESTINATION_POSITION, ModelAxis.SOURCE_POSITION
    ) == [(0, 1, last, key) for key in range(n_positions(small_model))]


def test_the_configured_activation_is_stamped_on_every_row(small_model, planned):
    index_df = planned(small_model, {"activation": "attn.out", "location": ["layer"]})

    assert values_of(index_df, ModelAxis.ACTIVATION) == [ModelLayerLocation.ATTENTION_OUT] * small_model.cfg.n_layers


def test_without_a_location_only_the_activation_is_left(small_model, planned):
    # The product over zero axes leaves a single empty row, so nothing but the stamped activation remains.
    from_empty = planned(small_model, {"activation": "resid.post", "location": []})
    from_missing = planned(small_model, {"activation": "resid.post"})

    assert list(from_empty.columns) == list(from_missing.columns) == [ModelAxis.ACTIVATION]
    assert len(from_empty) == len(from_missing) == 1


# --------------------------------------------------------------------------------------------------
# Filter - inclusion terms are OR-ed, keys within a term AND-ed, exclusions subtracted
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
def test_a_filter_narrows_a_single_axis(model, planned, filter_config, expected_layers):
    # GPT-2's 12 layers give the ranges and negative anchors above room to differ from each other.
    assert model.cfg.n_layers == 12

    index_df = planned(model, {"activation": "resid.post", "location": ["layer"], "filter": filter_config})

    assert values_of(index_df, ModelAxis.LAYER) == expected_layers


def test_the_keys_within_a_filter_term_are_and_ed(small_model, planned):
    index_df = planned(
        small_model, {"activation": "attn.z", "location": ["layer", "head"], "filter": [{"layer": 0, "head": 1}]}
    )

    assert coordinates_of(index_df, ModelAxis.LAYER, ModelAxis.HEAD) == [(0, 1)]


def test_filter_terms_are_or_ed_across_locations(small_model, planned):
    # layer 0 for any head, OR head 1 for any layer.
    index_df = planned(
        small_model, {"activation": "attn.z", "location": ["layer", "head"], "filter": [{"layer": 0}, {"head": 1}]}
    )

    assert set(coordinates_of(index_df, ModelAxis.LAYER, ModelAxis.HEAD)) == {
        *((0, head) for head in range(small_model.cfg.n_heads)),
        *((layer, 1) for layer in range(small_model.cfg.n_layers)),
    }


# --------------------------------------------------------------------------------------------------
# Transforms - group collapses values into one cell, incremental grows prefixes, combination expands
# a grouped cell again. They run in that order, after the filter.
# --------------------------------------------------------------------------------------------------
def _layers(planned, model, **source) -> list:
    return values_of(planned(model, {"activation": "resid.post", "location": ["layer"], **source}), ModelAxis.LAYER)


def test_group_collapses_the_listed_values_into_one_row(small_model, planned):
    # 0 and 1 collapse into one row addressing both; 2 is left untouched.
    assert _layers(planned, small_model, group={"layer": [0, 1]}) == [(0, 1), 2]


def test_group_all_collapses_the_whole_axis(small_model, planned):
    assert _layers(planned, small_model, group={"layer": "all"}) == [tuple(range(small_model.cfg.n_layers))]


def test_group_drops_the_duplicate_rows_it_creates(small_model, planned):
    # Grouping one axis leaves one row per value of the other, not one per pair.
    index_df = planned(small_model, {"activation": "attn.z", "location": ["layer", "head"], "group": {"layer": "all"}})

    assert coordinates_of(index_df, ModelAxis.LAYER, ModelAxis.HEAD) == [
        (tuple(range(small_model.cfg.n_layers)), head) for head in range(small_model.cfg.n_heads)
    ]


def test_group_counts_a_negative_value_from_the_end_of_its_axis(small_model, planned):
    def heads(values: list[int]) -> list:
        index_df = planned(small_model, {"activation": "attn.z", "location": ["head"], "group": {"head": values}})
        return values_of(index_df, ModelAxis.HEAD)

    n_heads = small_model.cfg.n_heads

    assert heads([-2, -1]) == heads([n_heads - 2, n_heads - 1]) == [*range(n_heads - 2), (n_heads - 2, n_heads - 1)]


def test_incremental_appends_growing_prefixes_to_every_row(model, planned):
    layers = _layers(planned, model, filter=[{"layer": [0, 1]}], incremental={"layer": [7, 8]})

    assert layers == [(0,), (0, 7), (0, 7, 8), (1,), (1, 7), (1, 7, 8)]


def test_incremental_can_add_a_location_the_index_does_not_have(small_model, planned):
    # The neuron axis is not in the index, so only the incremented rows exist.
    index_df = planned(
        small_model,
        {"activation": "mlp.post", "location": ["layer"], "filter": [{"layer": 0}], "incremental": {"neuron": [4, 5]}},
    )

    assert coordinates_of(index_df, ModelAxis.LAYER, ModelAxis.NEURON) == [(0, (4,)), (0, (4, 5))]


def test_incremental_counts_a_negative_value_from_the_end_of_its_axis(small_model, planned):
    def heads(values: list[int]) -> list:
        locator = {
            "activation": "attn.z",
            "location": ["head"],
            "filter": [{"head": 0}],
            "incremental": {"head": values},
        }
        return values_of(planned(small_model, locator), ModelAxis.HEAD)

    last = small_model.cfg.n_heads - 1

    assert heads([-1]) == heads([last]) == [(0,), (0, last)]


def test_incremental_keeps_the_other_locations(model, planned):
    index_df = planned(
        model,
        {
            "activation": "attn.z",
            "location": ["layer", "head"],
            "filter": [{"layer": 0, "head": [0, 1]}],
            "incremental": {"head": [9]},
        },
    )

    assert coordinates_of(index_df, ModelAxis.LAYER, ModelAxis.HEAD) == [
        (0, (0,)),
        (0, (0, 9)),
        (0, (1,)),
        (0, (1, 9)),
    ]


@pytest.mark.parametrize(
    ("combination", "expected"),
    [
        # max_elements caps the size of a combination...
        ({"location": ["layer"], "max_elements": 2}, [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]),
        # ...and without it every non-empty combination is produced.
        ({"location": ["layer"]}, [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]),
    ],
)
def test_combination_expands_a_grouped_axis(small_model, planned, combination, expected):
    assert small_model.cfg.n_layers == 3

    assert _layers(planned, small_model, group={"layer": "all"}, combination=combination) == expected


def test_filter_group_and_combination_run_in_that_order(model, planned):
    # The filter narrows the axis to 0..2 first, so the combinations never mention any later layer.
    layers = _layers(
        planned,
        model,
        filter=[{"layer": {"range": [0, 2]}}],
        group={"layer": "all"},
        combination={"location": ["layer"], "max_elements": 2},
    )

    assert layers == [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]


# --------------------------------------------------------------------------------------------------
# Explicit coordinates - the exact points, in the order they are listed
# --------------------------------------------------------------------------------------------------
def test_coordinates_are_indexed_in_order(model, planned):
    index_df = planned(
        model, {"activation": "resid.post", "coordinates": [{"layer": 2, "position": 1}, {"layer": 0, "position": 3}]}
    )

    assert coordinates_of(index_df, ModelAxis.LAYER, ModelAxis.POSITION) == [(2, 1), (0, 3)]
    assert values_of(index_df, ModelAxis.ACTIVATION) == [ModelLayerLocation.RESIDUAL_POST] * 2


def test_coordinates_resolve_negative_values(model, planned):
    index_df = planned(model, {"activation": "mlp.post", "coordinates": [{"layer": -1, "neuron": -2}]})

    assert coordinates_of(index_df, ModelAxis.LAYER, ModelAxis.NEURON) == [
        (model.cfg.n_layers - 1, model.cfg.d_mlp - 2)
    ]


def test_a_coordinate_list_addresses_several_values_in_one_row(model, planned):
    # A list value addresses those elements together in one row; negatives resolve inside lists too.
    index_df = planned(model, {"activation": "resid.post", "coordinates": [{"layer": 1, "position": [0, -1]}]})

    assert coordinates_of(index_df, ModelAxis.POSITION) == [((0, n_positions(model) - 1),)]


def test_coordinates_carry_their_own_activation(model, planned):
    index_df = planned(
        model, {"coordinates": [{"activation": "resid.post", "layer": 0}, {"activation": "mlp.out", "layer": -1}]}
    )

    # The activation column comes last, as index.csv writes it.
    assert list(index_df.columns) == [ModelAxis.LAYER, ModelAxis.ACTIVATION]
    assert values_of(index_df, ModelAxis.ACTIVATION) == [
        ModelLayerLocation.RESIDUAL_POST,
        ModelLayerLocation.MLP_OUT,
    ]
    assert values_of(index_df, ModelAxis.LAYER) == [0, model.cfg.n_layers - 1]


def test_a_negative_coordinate_resolves_against_its_own_activations_dimension(model, planned):
    # The same {neuron: -1} is d_head-1 for a head activation and d_mlp-1 for an MLP one.
    index_df = planned(
        model,
        {
            "coordinates": [
                {"activation": "attn.q", "layer": 0, "neuron": -1},
                {"activation": "mlp.post", "layer": 0, "neuron": -1},
            ]
        },
    )

    assert values_of(index_df, ModelAxis.NEURON) == [model.cfg.d_head - 1, model.cfg.d_mlp - 1]


def test_an_empty_coordinate_list_yields_no_rows(model, planned):
    index_df = planned(model, {"activation": "resid.post", "coordinates": []})

    assert len(index_df) == 0
    assert ModelAxis.ACTIVATION in index_df.columns


# --------------------------------------------------------------------------------------------------
# Built in code - the typed sections plan the index a config file's raw shapes do
# --------------------------------------------------------------------------------------------------
def test_a_typed_locator_built_in_code_plans_the_index_its_config_file_twin_does(model, planned):
    raw = {
        "activation": "attn.z",
        "location": ["layer", "head"],
        "filter": [{"layer": [0, 1]}, {"not": {"head": 0}}],
        "group": {"head": "all"},
        "incremental": {"neuron": [-1]},
        "combination": {"location": ["head"], "max_elements": 2},
    }
    typed = PatchingElementConfig(
        activation=ModelLayerLocation.ATTENTION_Z,
        location=[ModelAxis.LAYER, ModelAxis.HEAD],
        filter=Filter(inclusion_terms=[{ModelAxis.LAYER: [0, 1]}], exclusion_terms=[{ModelAxis.HEAD: 0}]),
        group=Group({ModelAxis.HEAD: "all"}),
        incremental=Incremental({ModelAxis.NEURON: [-1]}),
        combination=Combination(location=[ModelAxis.HEAD], max_elements=2),
        type=PatchingType.REPLACE,
    )

    from_file = planned(model, raw)

    pd.testing.assert_frame_equal(planned(model, typed), from_file)
    assert len(from_file) > 0


def test_a_code_built_coordinate_is_checked_too():
    # Built as ``Coordinates`` the deserializer's own checks are skipped, so the config repeats them.
    with pytest.raises(ValueError, match="Invalid coordinate"):
        PatchingElementConfig(
            activation=ModelLayerLocation.RESIDUAL_POST, coordinates=Coordinates([{}]), type=PatchingType.REPLACE
        )


# --------------------------------------------------------------------------------------------------
# Symbolic positions - a probe over prompts of different lengths keeps a position unresolved
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
def test_a_position_stays_symbolic_when_the_prompt_length_varies(
    small_model, planned_symbolic, filter_config, expected_positions
):
    index_df = planned_symbolic(
        small_model, {"activation": "resid.post", "location": ["position"], "filter": filter_config}
    )

    assert values_of(index_df, ModelAxis.POSITION) == expected_positions


def test_the_patterns_positions_stay_symbolic_when_the_prompt_length_varies(small_model, planned_symbolic):
    index_df = planned_symbolic(
        small_model,
        {
            "activation": "attn.pattern",
            "location": ["layer", "head", "dest_pos", "src_pos"],
            "filter": [{"layer": 0, "head": 0, "dest_pos": -1, "src_pos": [0, -2]}],
        },
    )

    assert coordinates_of(index_df, ModelAxis.DESTINATION_POSITION, ModelAxis.SOURCE_POSITION) == [(-1, -2), (-1, 0)]


def test_a_symbolic_position_keeps_the_other_locations(small_model, planned_symbolic):
    index_df = planned_symbolic(
        small_model, {"activation": "resid.post", "location": ["layer", "position"], "filter": [{"position": -1}]}
    )

    assert coordinates_of(index_df, ModelAxis.LAYER, ModelAxis.POSITION) == [
        (layer, -1) for layer in range(small_model.cfg.n_layers)
    ]


def test_a_known_prompt_length_resolves_a_negative_position(small_model, planned):
    # The contrast: with a length to count against, -1 collapses to its absolute index.
    index_df = planned(
        small_model, {"activation": "resid.post", "location": ["position"], "filter": [{"position": -1}]}
    )

    assert values_of(index_df, ModelAxis.POSITION) == [n_positions(small_model) - 1]


def test_symbolic_coordinates_keep_their_signs(small_model, planned_symbolic):
    index_df = planned_symbolic(
        small_model,
        {"activation": "resid.post", "coordinates": [{"layer": 1, "position": -2}, {"layer": 0, "position": [0, -1]}]},
    )

    assert coordinates_of(index_df, ModelAxis.LAYER, ModelAxis.POSITION) == [(1, -2), (0, (0, -1))]


def test_a_transform_treats_a_symbolic_position_like_any_other_value(small_model, planned_symbolic):
    # Group collapses negative positions just as it collapses non-negative ones.
    index_df = planned_symbolic(
        small_model,
        {
            "activation": "resid.post",
            "location": ["position"],
            "filter": [{"position": {"range": [-3, -1]}}],
            "group": {"position": [-2, -1]},
        },
    )

    assert values_of(index_df, ModelAxis.POSITION) == [-3, (-2, -1)]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        # an unfiltered position axis would enumerate every position
        ({"location": ["position"]}, "prompt length varies"),
        # an exclusion still needs the full range to subtract from
        ({"location": ["position"], "filter": [{"not": {"position": 0}}]}, "prompt length varies"),
        # a mixed-sign range has no length-independent meaning
        ({"location": ["position"], "filter": [{"position": {"range": [-1, 1]}}]}, "positions on the same side"),
        # the attention pattern's positions are sized by the prompt too
        ({"activation": "attn.pattern", "location": ["src_pos"]}, "Cannot build a grid over 'src_pos'"),
    ],
)
def test_a_position_grid_that_depends_on_the_prompt_length_raises(small_model, planned_symbolic, source, message):
    with pytest.raises(ValueError, match=message):
        planned_symbolic(small_model, {"activation": "resid.post", **source})


# --------------------------------------------------------------------------------------------------
# Rejected configurations
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("source", "message"),
    [
        ({"filter": [{}]}, r"Invalid filter term: \{\}"),  # empty pattern
        ({"filter": [{"not": {}}]}, "Invalid filter term: {'not': {}}"),  # empty negated pattern
        ({"filter": ["foo"]}, "Invalid filter term: 'foo'"),  # term is not a mapping
        ({"filter": [5]}, "Invalid filter term: 5"),
        ({"location": ["bogus"]}, "'bogus' is not a valid ModelAxis"),  # not a ModelAxis value
        ({"fitler": [{"layer": 0}]}, r"Unknown keys for PatchingElementConfig: \['fitler'\]"),  # a typo is rejected
        ({"filter": [{"bogus": 0}]}, "'bogus' is not a valid ModelAxis"),
        ({"group": {"bogus": "all"}}, "'bogus' is not a valid ModelAxis"),
        # a section that keys values by location needs a mapping
        ({"group": "all"}, "Expected a mapping for Group, got 'all'"),
        ({"incremental": [1, 2]}, r"Expected a mapping for Incremental, got \[1, 2\]"),
        ({"location": ["activation"]}, "'activation' is not a declarative location"),
        ({"coordinates": [{"bogus": 0}]}, "'bogus' is not a valid ModelAxis"),
        # the two modes are exclusive
        ({"coordinates": [{"layer": 0}], "location": ["layer"]}, "'coordinates' cannot be combined with 'location'"),
        ({"coordinates": [{"layer": 0}], "filter": [{"layer": 0}]}, "'coordinates' cannot be combined with"),
        # coordinates do not share their locations
        ({"coordinates": [{"layer": 0}, {"head": 0}]}, "All coordinates must share the same keys"),
        ({"coordinates": [{}]}, r"Invalid coordinate: \{\}"),  # empty coordinate
        # coordinates are a list of mappings, not a scalar
        ({"coordinates": "layer"}, "Expected a list of coordinates, got 'layer'"),
        ({"coordinates": [{"activation": "bogus", "layer": 0}]}, "'bogus' is not a valid ModelLayerLocation"),
        (
            {"activation": "resid.post", "coordinates": [{"activation": "resid.post", "layer": 0}]},
            "'activation' cannot be set both on the config and inside the coordinates",
        ),
        # only some coordinates carry an activation
        ({"coordinates": [{"activation": "resid.post", "layer": 0}, {"layer": 1}]}, "must share the same keys"),
        # a location with nothing to read it in
        ({"activation": None, "location": ["layer"]}, "names a location but no 'activation'"),
        ({"activation": None, "filter": [{"layer": 0}]}, "names a location but no 'activation'"),
        ({"coordinates": [{"layer": 0}]}, "names a location but no 'activation'"),
        # an axis the activation's tensor does not have, which a run would silently read whole
        ({"activation": "attn.pattern", "location": ["layer", "head", "position"]}, "'attn.pattern' has no 'position'"),
        ({"activation": "attn.pattern", "location": ["layer", "neuron"]}, "'attn.pattern' has no 'neuron'"),
        ({"location": ["layer", "dest_pos"]}, "'resid.post' has no 'dest_pos'"),
        ({"activation": "attn.z", "location": ["layer", "src_pos"]}, "'attn.z' has no 'src_pos'"),
        ({"location": ["layer", "head"]}, "'resid.post' has no 'head'"),
        ({"activation": "attn.in", "location": ["layer", "head"]}, "'attn.in' has no 'head'"),
        ({"location": ["layer"], "incremental": {"head": [0]}}, "'resid.post' has no 'head'"),
        ({"activation": "embed", "location": ["layer", "position"]}, "'embed' has no 'layer'"),
        (
            {"coordinates": [{"activation": "attn.pattern", "layer": 0, "position": 0}]},
            "'attn.pattern' has no 'position'",
        ),
        # a transform reshaping an axis the index does not enumerate
        (
            {"activation": "attn.z", "location": ["layer", "position"], "group": {"head": [0, 1]}},
            "'group' merges 'head', which 'location' does not name",
        ),
        (
            {"activation": "attn.z", "location": ["layer"], "combination": {"location": ["head"]}},
            "'combination' expands 'head', which neither 'location' nor 'incremental' names",
        ),
    ],
)
def test_a_malformed_locator_is_rejected_when_its_config_is_loaded(config_from_yaml, source, message):
    # Coordinates carry their own activation, so only the declarative shapes get the config-level one.
    default = {} if "coordinates" in source else {"activation": "resid.post"}
    element = {**default, **source, "type": "replace"}

    with pytest.raises(ValueError, match=message):
        config_from_yaml(PatchingConfig, "broken", {**PATCHING, "patch": [{"source": element, "target": element}]})


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ({"location": ["layer"], "filter": [{"layer": "x"}]}, "Invalid filter constraint for 'layer': 'x'"),
        # the filter names a location the index does not have
        ({"location": ["layer"], "filter": [{"head": 0}]}, "Filter pattern references unknown locations"),
        ({"coordinates": [{"layer": "x"}]}, "Invalid coordinate value for 'layer': 'x'"),
        # a value past either end of its axis names no row the model has
        ({"location": ["layer"], "filter": [{"layer": 3}]}, "'layer' 3 is out of range for an axis of 3"),
        ({"location": ["layer"], "filter": [{"layer": -4}]}, "'layer' -4 is out of range for an axis of 3"),
        ({"location": ["layer"], "filter": [{"layer": {"range": [1, 3]}}]}, "'layer' 3 is out of range"),
        ({"location": ["layer"], "filter": [{"not": {"layer": 3}}]}, "'layer' 3 is out of range"),
        (
            {"location": ["layer", "neuron"], "filter": [{"neuron": 16}]},
            "'neuron' 16 is out of range for an axis of 16",
        ),
        ({"location": ["position"], "filter": [{"position": 6}]}, "'position' 6 is out of range for an axis of 6"),
        (
            {"activation": "attn.pattern", "location": ["src_pos"], "filter": [{"src_pos": 6}]},
            "'src_pos' 6 is out of range for an axis of 6",
        ),
        ({"coordinates": [{"layer": 3}]}, "'layer' 3 is out of range"),
        ({"coordinates": [{"layer": 0, "position": [0, -7]}]}, "'position' -7 is out of range"),
        ({"activation": "attn.z", "location": ["head"], "group": {"head": [0, 4]}}, "'head' 4 is out of range"),
        (
            {"activation": "mlp.post", "location": ["layer"], "incremental": {"neuron": [6]}},
            "'neuron' 6 is out of range",
        ),
    ],
)
def test_a_locator_the_model_cannot_resolve_raises_when_it_is_planned(small_model, planned, source, message):
    # The config loads; it is resolving it against a model that fails, before anything runs.
    with pytest.raises(ValueError, match=message):
        planned(small_model, {"activation": "resid.post", **source})
