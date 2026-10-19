"""What a probe is trained on, read back from the probe the run returns.

A trained ``LogisticRegression`` carries one coefficient per feature, so the shape of ``coef_`` is
the feature size a pooling config produces, and refitting on features reduced by hand reproduces the
run's coefficients exactly when - and only when - the run pooled the same way.
"""

import pytest
import torch

from entoscope.model.config import Filter, Group, ModelAxis, ModelLayerLocation
from entoscope.probe.config import PoolingType, ProbeConfig
from entoscope.probe.input import ProbeInput
from entoscope.probe.prober import Prober

from ..conftest import forbid_model_runs
from .conftest import (
    NEGATIVE,
    POSITIVE,
    SHORT_POSITIVE,
    activation_caches,
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
    collector = run_probe(small_model, probe_config(pool={ModelAxis.POSITION: pooling}))

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
            pool={ModelAxis.POSITION: PoolingType.MEAN, ModelAxis.HEAD: PoolingType.MAX},
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

    position_first = collector_for({ModelAxis.POSITION: PoolingType.MEAN, ModelAxis.HEAD: PoolingType.MAX})
    head_first = collector_for({ModelAxis.HEAD: PoolingType.MAX, ModelAxis.POSITION: PoolingType.MEAN})

    assert_trained_alike(probe_of(position_first), probe_of(head_first))


def test_a_pooling_entry_that_matches_no_axis_is_ignored(small_model):
    # The residual stream has no head axis, and the grid already pinned the layer.
    pooled = run_probe(small_model, probe_config(pool={ModelAxis.POSITION: PoolingType.MEAN}))
    with_extra = run_probe(
        small_model, probe_config(pool={ModelAxis.POSITION: PoolingType.MEAN, ModelAxis.HEAD: PoolingType.MAX})
    )

    assert_trained_alike(probe_of(pooled), probe_of(with_extra))


def test_both_of_the_patterns_positions_are_pooled_by_name(small_model, small_caches):
    # attn.pattern is [head, dest_pos, src_pos]: pooling both positions, with the grid pinning the
    # head, leaves a single feature.
    collector = run_probe(
        small_model,
        probe_config(
            activation=ModelLayerLocation.ATTENTION_PATTERN,
            location=[ModelAxis.LAYER, ModelAxis.HEAD],
            pool={ModelAxis.DESTINATION_POSITION: PoolingType.SUM, ModelAxis.SOURCE_POSITION: PoolingType.SUM},
        ),
    )

    assert probe_of(collector, head=0).coef_.shape[1] == 1
    expected = _stacked(small_caches, "blocks.0.attn.hook_pattern", lambda p: p[0].sum(dim=0).sum(dim=0).reshape(1))
    assert_trained_alike(probe_of(collector, head=0), refit(expected))


def test_position_is_no_axis_of_the_pattern_to_pool(small_model):
    # The pattern's positions go by their own names, so a 'position' entry matches neither of them and
    # is ignored like any entry that matches no axis: every query and key position stays a feature.
    collector = run_probe(
        small_model,
        probe_config(
            activation=ModelLayerLocation.ATTENTION_PATTERN,
            location=[ModelAxis.LAYER, ModelAxis.HEAD],
            pool={ModelAxis.POSITION: PoolingType.SUM},
        ),
        positive=EQUAL_POSITIVE,
        negative=EQUAL_NEGATIVE,
    )

    positions = small_model.to_tokens(EQUAL_POSITIVE[0]).shape[1]
    assert probe_of(collector, head=0).coef_.shape[1] == positions * positions


def test_one_of_the_patterns_positions_is_pooled_while_the_other_is_pinned(small_model, small_caches):
    # The last token's largest attention weight in each prompt: dest_pos -1 is read from each prompt's
    # own end, and src_pos is reduced over however many keys that prompt has.
    collector = run_probe(
        small_model,
        probe_config(
            activation=ModelLayerLocation.ATTENTION_PATTERN,
            location=[ModelAxis.LAYER, ModelAxis.HEAD, ModelAxis.DESTINATION_POSITION],
            pool={ModelAxis.SOURCE_POSITION: PoolingType.MAX},
            filter=Filter.from_config([{"dest_pos": -1}]),
        ),
    )

    assert probe_of(collector, head=1).coef_.shape[1] == 1
    expected = _stacked(small_caches, "blocks.0.attn.hook_pattern", lambda p: p[1, -1].amax().reshape(1))
    assert_trained_alike(probe_of(collector, head=1), refit(expected))


def test_an_entry_of_the_pattern_is_read_against_each_prompts_own_end(small_model, small_caches):
    # How much the last token attends to the first, in prompts of different lengths.
    collector = run_probe(
        small_model,
        probe_config(
            activation=ModelLayerLocation.ATTENTION_PATTERN,
            location=[ModelAxis.LAYER, ModelAxis.HEAD, ModelAxis.DESTINATION_POSITION, ModelAxis.SOURCE_POSITION],
            pool={},
            filter=Filter.from_config([{"dest_pos": -1, "src_pos": 0}]),
        ),
    )

    assert probe_of(collector, head=2).coef_.shape[1] == 1
    expected = _stacked(small_caches, "blocks.0.attn.hook_pattern", lambda p: p[2, -1, 0].reshape(1))
    assert_trained_alike(probe_of(collector, head=2), refit(expected))


def test_a_pooling_type_that_is_not_known_raises():
    # A config built in code is read as a file's is, so a pooling nothing implements is refused as it is built.
    with pytest.raises(ValueError, match="'median' is not a valid PoolingType"):
        probe_config(pool={ModelAxis.POSITION: "median"})


def test_a_single_value_becomes_a_single_feature(small_model, small_caches):
    # One neuron at one position leaves nothing to flatten - the activation is a scalar.
    config = probe_config(
        activation=ModelLayerLocation.MLP_POST,
        location=[ModelAxis.LAYER, ModelAxis.NEURON, ModelAxis.POSITION],
        pool={},
        filter=Filter.from_config([{"neuron": 1, "position": -1}]),
    )

    collector = run_probe(small_model, config)

    assert _feature_size(collector) == 1
    expected = _stacked(small_caches, "blocks.0.mlp.hook_post", lambda post: post[-1, 1].reshape(1))
    assert_trained_alike(probe_of(collector), refit(expected))


def test_two_grouped_axes_read_every_combination_of_their_values(small_model, small_caches):
    # Grouping every head and every head dimension lists both axes in one row, which reads the whole
    # block they span: every head's output, where pairing the two lists would keep only the diagonal.
    collector = run_probe(
        small_model,
        probe_config(
            activation=ModelLayerLocation.ATTENTION_Z,
            location=[ModelAxis.LAYER, ModelAxis.HEAD, ModelAxis.NEURON],
            group=Group({ModelAxis.HEAD: "all", ModelAxis.NEURON: "all"}),
        ),
    )

    assert _feature_size(collector) == small_model.cfg.n_heads * small_model.cfg.d_head
    expected = _stacked(small_caches, "blocks.0.attn.hook_z", lambda z: z.mean(dim=0).flatten())
    assert_trained_alike(probe_of(collector), refit(expected))


def test_an_axis_read_whole_between_two_grouped_ones_is_read_at_each_prompts_own_length(small_model, small_caches):
    # Two heads' weights on the last and the first key, averaged over however many queries each prompt has.
    collector = run_probe(
        small_model,
        probe_config(
            activation=ModelLayerLocation.ATTENTION_PATTERN,
            location=[ModelAxis.LAYER, ModelAxis.HEAD, ModelAxis.SOURCE_POSITION],
            pool={ModelAxis.DESTINATION_POSITION: PoolingType.MEAN},
            filter=Filter.from_config([{"head": [0, 1], "src_pos": [-1, 0]}]),
            group=Group({ModelAxis.HEAD: "all", ModelAxis.SOURCE_POSITION: "all"}),
        ),
    )

    assert _feature_size(collector) == 4
    expected = _stacked(
        small_caches, "blocks.0.attn.hook_pattern", lambda p: p[[0, 1]][:, :, [-1, 0]].mean(dim=1).flatten()
    )
    assert_trained_alike(probe_of(collector), refit(expected))


def test_an_unpooled_pattern_needs_prompts_of_equal_length(small_model):
    config = probe_config(
        activation=ModelLayerLocation.ATTENTION_PATTERN,
        location=[ModelAxis.LAYER, ModelAxis.HEAD],
        pool={},
    )

    with pytest.raises(ValueError, match="Feature size varies"):
        run_probe(small_model, config)


# --------------------------------------------------------------------------------------------------
# Positions - a filtered position is sliced against each prompt's own cache
# --------------------------------------------------------------------------------------------------
def _position_config(*positions: int) -> ProbeConfig:
    return probe_config(
        location=[ModelAxis.LAYER, ModelAxis.POSITION],
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
    from_start = [single.output for single in collector.operation_input.index if single.source[ModelAxis.POSITION] == 2]
    expected = _stacked(small_caches, "blocks.0.hook_resid_post", lambda residual: residual[2])
    assert_trained_alike(collector.probes[from_start[0]], refit(expected))


@pytest.mark.parametrize("position", [40, -40])
def test_a_position_outside_the_shortest_prompt_raises(small_model, position):
    with pytest.raises(ValueError, match="out of range for the shortest prompt"):
        run_probe(small_model, _position_config(position))


@pytest.mark.parametrize("position", [40, -40])
def test_a_pattern_position_outside_the_shortest_prompt_raises(small_model, monkeypatch, position):
    config = probe_config(
        activation=ModelLayerLocation.ATTENTION_PATTERN,
        location=[ModelAxis.LAYER, ModelAxis.HEAD, ModelAxis.DESTINATION_POSITION, ModelAxis.SOURCE_POSITION],
        pool={},
        filter=Filter.from_config([{"dest_pos": -1, "src_pos": position}]),
    )
    # Refused as the index is resolved, before the first prompt is run.
    forbid_model_runs(monkeypatch, small_model)

    with pytest.raises(ValueError, match=f"'src_pos' {position} is out of range for the shortest prompt"):
        Prober(small_model).plan(ProbeInput(config, POSITIVE, NEGATIVE), verbose=False)


def test_a_full_position_grid_is_built_for_prompts_of_equal_length(small_model):
    collector = run_probe(
        small_model,
        probe_config(location=[ModelAxis.LAYER, ModelAxis.POSITION]),
        positive=EQUAL_POSITIVE,
        negative=EQUAL_NEGATIVE,
    )

    positions = small_model.to_tokens(EQUAL_POSITIVE[0]).shape[1]
    assert len(collector.result) == small_model.cfg.n_layers * positions
    assert _feature_size(collector) == small_model.cfg.d_model


def test_shorter_and_longer_prompts_share_one_grid(small_model):
    # The positive prompts are labelled 1 and the negative ones 0 whatever their length, so the probe is
    # the one trained on the shorter prompts' last residuals against the longer prompts'.
    collector = run_probe(small_model, _position_config(-1), positive=SHORT_POSITIVE, negative=POSITIVE)

    assert len(collector.result) == small_model.cfg.n_layers
    caches = activation_caches(small_model, SHORT_POSITIVE + POSITIVE)
    expected = _stacked(caches, "blocks.0.hook_resid_post", lambda residual: residual[-1])
    assert_trained_alike(probe_of(collector), refit(expected, labels=[1] * len(SHORT_POSITIVE) + [0] * len(POSITIVE)))


def test_a_run_is_reproducible(small_model):
    first = run_probe(small_model, probe_config()).result
    second = run_probe(small_model, probe_config()).result

    assert first.equals(second)
