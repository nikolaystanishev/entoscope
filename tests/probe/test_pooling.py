"""What a probe is trained on, read back from the probe the run returns.

A trained ``LogisticRegression`` carries one coefficient per feature, so the shape of ``coef_`` is
the feature size a pooling config produces, and refitting on features reduced by hand reproduces the
run's coefficients exactly when — and only when — the run pooled the same way.
"""

import pytest
import torch

from entoscope.model.config import Filter, Group, ModelLayerLocation, ModelLocation
from entoscope.probe.config import PoolingType, ProbeConfig
from entoscope.probe.input import ProbeInput
from entoscope.probe.prober import Prober

from .conftest import (
    NEGATIVE,
    POSITIVE,
    SHORT_POSITIVE,
    assert_trained_alike,
    probe_config,
    probe_of,
    refit,
    run_probe,
)

EQUAL_POSITIVE = ["The movie was wonderful"]
EQUAL_NEGATIVE = ["The movie was dreadful"]


def _feature_size(collector, layer: int = 0) -> int:
    return probe_of(collector, layer).coef_.shape[1]


def _stacked(caches, name: str, reduce) -> torch.Tensor:
    return torch.stack([reduce(cache[name][0]) for cache in caches])


# --------------------------------------------------------------------------------------------------
# Pooling reduces the axes it names and flattens the rest
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("pooling", "reduce"),
    [
        (PoolingType.MEAN, lambda activation: activation.mean(dim=0)),
        (PoolingType.MAX, lambda activation: activation.amax(dim=0)),
        (PoolingType.SUM, lambda activation: activation.sum(dim=0)),
        (PoolingType.LAST, lambda activation: activation[-1]),
    ],
)
def test_a_pooled_position_axis_is_reduced_over_the_prompt(small_model, small_caches, pooling, reduce):
    collector = run_probe(small_model, probe_config(pool={ModelLocation.POSITION: pooling}))

    assert _feature_size(collector) == small_model.cfg.d_model
    assert_trained_alike(probe_of(collector), refit(_stacked(small_caches, "blocks.0.hook_resid_post", reduce)))


def test_without_pooling_every_position_becomes_a_feature(small_model):
    collector = run_probe(small_model, probe_config(pool={}), positive=EQUAL_POSITIVE, negative=EQUAL_NEGATIVE)

    positions = small_model.to_tokens(EQUAL_POSITIVE[0]).shape[1]
    assert _feature_size(collector) == positions * small_model.cfg.d_model


def test_unpooled_positions_need_prompts_of_equal_length(small_model):
    # Without pooling the feature vector carries every position, so its size follows the prompt.
    with pytest.raises(ValueError, match="Feature size varies"):
        run_probe(small_model, probe_config(pool={}))


def test_head_features_survive_position_pooling(small_model):
    collector = run_probe(small_model, probe_config(activation=ModelLayerLocation.ATTENTION_Z))

    assert _feature_size(collector) == small_model.cfg.n_heads * small_model.cfg.d_head


def test_each_axis_is_pooled_with_its_own_type(small_model, small_caches):
    collector = run_probe(
        small_model,
        probe_config(
            activation=ModelLayerLocation.ATTENTION_Z,
            pool={ModelLocation.POSITION: PoolingType.MEAN, ModelLocation.HEAD: PoolingType.MAX},
        ),
    )

    assert _feature_size(collector) == small_model.cfg.d_head
    expected = _stacked(small_caches, "blocks.0.attn.hook_z", lambda z: z.mean(dim=0).amax(dim=0))
    assert_trained_alike(probe_of(collector), refit(expected))


def test_axes_are_pooled_in_their_own_order_not_the_configs(small_model):
    # Position sits before head on the activation, so it is reduced first either way; the reverse
    # order (max over heads, then mean) would give different features.
    def collector_for(pool: dict):
        return run_probe(small_model, probe_config(activation=ModelLayerLocation.ATTENTION_Z, pool=pool))

    position_first = collector_for({ModelLocation.POSITION: PoolingType.MEAN, ModelLocation.HEAD: PoolingType.MAX})
    head_first = collector_for({ModelLocation.HEAD: PoolingType.MAX, ModelLocation.POSITION: PoolingType.MEAN})

    assert_trained_alike(probe_of(position_first), probe_of(head_first))


def test_a_pooling_entry_that_matches_no_axis_is_ignored(small_model):
    # The residual stream has no head axis, and the grid already pinned the layer.
    pooled = run_probe(small_model, probe_config(pool={ModelLocation.POSITION: PoolingType.MEAN}))
    with_extra = run_probe(
        small_model, probe_config(pool={ModelLocation.POSITION: PoolingType.MEAN, ModelLocation.HEAD: PoolingType.MAX})
    )

    assert_trained_alike(probe_of(pooled), probe_of(with_extra))


def test_both_of_the_patterns_position_axes_are_pooled(small_model, small_caches):
    # attn.pattern is [head, dest_pos, src_pos]: pooling 'position' reduces both, and the grid has
    # already pinned the head, so a single feature is left.
    collector = run_probe(
        small_model,
        probe_config(
            activation=ModelLayerLocation.ATTENTION_PATTERN,
            location=[ModelLocation.LAYER, ModelLocation.HEAD],
            pool={ModelLocation.POSITION: PoolingType.SUM},
        ),
    )

    assert probe_of(collector, head=0).coef_.shape[1] == 1
    expected = _stacked(small_caches, "blocks.0.attn.hook_pattern", lambda p: p[0].sum(dim=0).sum(dim=0).reshape(1))
    assert_trained_alike(probe_of(collector, head=0), refit(expected))


def test_a_pooling_type_that_is_not_known_raises(small_model):
    with pytest.raises(ValueError, match="Unknown pooling type"):
        run_probe(small_model, probe_config(pool={ModelLocation.POSITION: "median"}))


def test_a_single_value_becomes_a_single_feature(small_model, small_caches):
    # One neuron at one position leaves nothing to flatten — the activation is a scalar.
    config = probe_config(
        activation=ModelLayerLocation.MLP_POST,
        location=[ModelLocation.LAYER, ModelLocation.NEURON, ModelLocation.POSITION],
        pool={},
        filter=Filter.from_config([{"neuron": 1, "position": -1}]),
    )

    collector = run_probe(small_model, config)

    assert _feature_size(collector) == 1
    expected = _stacked(small_caches, "blocks.0.mlp.hook_post", lambda post: post[-1, 1].reshape(1))
    assert_trained_alike(probe_of(collector), refit(expected))


def test_two_grouped_axes_cannot_be_named(small_model):
    # Grouping the head and the neuron axis leaves two list-indexed axes on one activation, which
    # merge into a single axis whose name is no longer attributable.
    config = probe_config(
        activation=ModelLayerLocation.ATTENTION_Z,
        location=[ModelLocation.LAYER, ModelLocation.HEAD, ModelLocation.NEURON],
        group=Group({ModelLocation.HEAD: "all", ModelLocation.NEURON: "all"}),
    )

    with pytest.raises(ValueError, match="more than one axis is indexed with a list"):
        run_probe(small_model, config)


def test_an_unpooled_pattern_needs_prompts_of_equal_length(small_model):
    config = probe_config(
        activation=ModelLayerLocation.ATTENTION_PATTERN,
        location=[ModelLocation.LAYER, ModelLocation.HEAD],
        pool={},
    )

    with pytest.raises(ValueError, match="Feature size varies"):
        run_probe(small_model, config)


# --------------------------------------------------------------------------------------------------
# Positions — a filtered position is sliced against each prompt's own cache
# --------------------------------------------------------------------------------------------------
def _position_config(*positions: int) -> ProbeConfig:
    return probe_config(
        location=[ModelLocation.LAYER, ModelLocation.POSITION],
        pool={},
        filter=Filter.from_config([{"position": position} for position in positions]),
    )


def test_the_last_position_is_read_from_each_prompts_own_end(small_model, small_caches):
    collector = run_probe(small_model, _position_config(-1))

    assert len(collector.result) == small_model.cfg.n_layers
    assert _feature_size(collector) == small_model.cfg.d_model
    expected = _stacked(small_caches, "blocks.0.hook_resid_post", lambda residual: residual[-1])
    assert_trained_alike(probe_of(collector), refit(expected))


def test_mixed_anchors_are_distinct_grid_points(small_model):
    collector = run_probe(small_model, _position_config(0, -1))
    result = collector.result

    assert len(result) == 2 * small_model.cfg.n_layers
    assert set(result["source_position"]) == {0, -1}


def test_an_absolute_and_a_relative_position_stay_apart(small_model, small_caches):
    # 2 counts from the start and -2 from the end; they collide on a four-token prompt but are
    # still two points, each sliced against its own prompt.
    collector = run_probe(small_model, _position_config(2, -2))
    result = collector.result

    assert set(result["source_position"]) == {2, -2}
    from_start = [
        single.output for single in collector.operation_input.index if single.source[ModelLocation.POSITION] == 2
    ]
    expected = _stacked(small_caches, "blocks.0.hook_resid_post", lambda residual: residual[2])
    assert_trained_alike(collector.probes[from_start[0]], refit(expected))


@pytest.mark.parametrize("position", [40, -40])
def test_a_position_outside_the_shortest_prompt_raises(small_model, position):
    with pytest.raises(ValueError, match="out of range for the shortest prompt"):
        run_probe(small_model, _position_config(position))


def test_a_full_position_grid_is_built_for_prompts_of_equal_length(small_model):
    collector = run_probe(
        small_model,
        probe_config(location=[ModelLocation.LAYER, ModelLocation.POSITION]),
        positive=EQUAL_POSITIVE,
        negative=EQUAL_NEGATIVE,
    )

    positions = small_model.to_tokens(EQUAL_POSITIVE[0]).shape[1]
    assert len(collector.result) == small_model.cfg.n_layers * positions
    assert _feature_size(collector) == small_model.cfg.d_model


def test_shorter_and_longer_prompts_share_one_grid(small_model):
    # The label order is positive prompts first, then negative ones, whatever their length.
    collector = run_probe(small_model, _position_config(-1), positive=SHORT_POSITIVE, negative=POSITIVE)

    assert len(collector.result) == small_model.cfg.n_layers
    assert collector.operation_input.labels.tolist() == [1, 1, 0, 0]


def test_a_run_is_reproducible(small_model):
    first = run_probe(small_model, probe_config()).result
    second = run_probe(small_model, probe_config()).result

    assert first.equals(second)


def test_the_prepared_input_carries_one_feature_matrix_per_location(small_model):
    # The feature matrices themselves never reach the collector, so this reads the prepared input —
    # the real model, the real index, nothing stubbed.
    probe_input = ProbeInput(probe_config(), POSITIVE, NEGATIVE).prepare(small_model)

    assert probe_input.labels.tolist() == [1, 1, 0, 0]
    assert set(probe_input.features) == {single.output for single in probe_input.index}
    for features in probe_input.features.values():
        assert features.shape == (len(POSITIVE) + len(NEGATIVE), small_model.cfg.d_model)
        assert features.dtype == torch.float32
        assert features.device.type == "cpu"

    # Preparing again keeps the same index instead of running the model a second time.
    assert probe_input.prepare(small_model).index is probe_input.index


def test_a_prepared_input_moves_to_a_device(small_model):
    probe_input = ProbeInput(probe_config(), POSITIVE, NEGATIVE).prepare(small_model)

    moved = probe_input.to("cpu")

    assert moved is probe_input
    assert moved.labels.device.type == "cpu"
    assert all(features.device.type == "cpu" for features in moved.features.values())


def test_an_unprepared_input_has_nothing_to_read(small_model):
    probe_input = ProbeInput(probe_config(), POSITIVE, NEGATIVE)

    for attribute in ("labels", "features", "index"):
        with pytest.raises(AssertionError):
            getattr(probe_input, attribute)

    Prober(small_model).apply(probe_input, verbose=False)
    assert probe_input.index is not None


def test_the_run_only_caches_the_activations_its_index_names(small_model):
    probe_input = ProbeInput(probe_config(), POSITIVE, NEGATIVE).prepare(small_model)

    assert probe_input.index.source_activation_names() == {
        f"blocks.{layer}.hook_resid_post" for layer in range(small_model.cfg.n_layers)
    }
