import pytest
import torch

from intex.model.config import Filter, ModelLayerLocation, ModelLocation, ModelLocatorConfig
from intex.probe.config import PoolingType, ProbeConfig, ProbeMetric, ProbeType
from intex.probe.input import ProbeInput

POSITIVE = ["The movie was absolutely wonderful", "I loved it"]
NEGATIVE = ["The movie was dreadful and boring", "I hated it"]

EQUAL_POSITIVE = ["The movie was great"]
EQUAL_NEGATIVE = ["The movie was bad"]


def _config(
    pool: dict[ModelLocation, PoolingType] | None = None,
    activation: ModelLayerLocation = ModelLayerLocation.RESIDUAL_POST,
    location: list[ModelLocation] | None = None,
) -> ProbeConfig:
    if pool is None:
        pool = {ModelLocation.POSITION: PoolingType.MEAN}
    return ProbeConfig(
        type=ProbeType.LINEAR,
        pool=pool,
        metric=[ProbeMetric.ACCURACY],
        source=ModelLocatorConfig(activation=activation, location=location or [ModelLocation.LAYER]),
    )


def test_prepare_populates_labels_features_and_index(model):
    probe_input = ProbeInput(_config(), POSITIVE, NEGATIVE).prepare(model)

    assert probe_input.labels.tolist() == [1, 1, 0, 0]
    assert probe_input.index.axis_range == [model.cfg.n_layers]
    assert set(probe_input.features) == {single.output for single in probe_input.index}
    for features in probe_input.features.values():
        assert features.shape == (4, model.cfg.d_model)
        assert features.dtype == torch.float32
        assert features.device.type == "cpu"


def test_prepare_is_idempotent(model):
    probe_input = ProbeInput(_config(), POSITIVE, NEGATIVE).prepare(model)
    index = probe_input.index
    probe_input.prepare(model)
    assert probe_input.index is index


def test_unprepared_access_raises():
    probe_input = ProbeInput(_config(), POSITIVE, NEGATIVE)
    for attribute in ("labels", "features", "index"):
        with pytest.raises(AssertionError):
            getattr(probe_input, attribute)


def test_mean_pooled_features_match_manual_cache_reduction(model):
    probe_input = ProbeInput(_config(), POSITIVE, NEGATIVE).prepare(model)

    tokens = model.to_tokens(POSITIVE[0])
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    expected = cache["blocks.0.hook_resid_post"][0].mean(dim=0).to("cpu", torch.float32)

    layer_output = next(single.output for single in probe_input.index if single.source.layer() == 0)
    assert torch.allclose(probe_input.features[layer_output][0], expected, atol=1e-5)


def test_no_pooling_flattens_all_positions(model):
    probe_input = ProbeInput(_config(pool={}), EQUAL_POSITIVE, EQUAL_NEGATIVE).prepare(model)

    positions = model.to_tokens(EQUAL_POSITIVE[0]).shape[1]
    for features in probe_input.features.values():
        assert features.shape == (2, positions * model.cfg.d_model)

    tokens = model.to_tokens(EQUAL_POSITIVE[0])
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    expected = cache["blocks.0.hook_resid_post"][0].flatten().to("cpu", torch.float32)
    layer_output = next(single.output for single in probe_input.index if single.source.layer() == 0)
    assert torch.allclose(probe_input.features[layer_output][0], expected, atol=1e-5)


def test_no_pooling_with_unequal_length_prompts_raises(model):
    with pytest.raises(ValueError, match="Feature size varies"):
        ProbeInput(_config(pool={}), POSITIVE, NEGATIVE).prepare(model)


def test_position_pooling_keeps_head_features(model):
    config = _config(activation=ModelLayerLocation.ATTENTION_Z)
    probe_input = ProbeInput(config, POSITIVE, NEGATIVE).prepare(model)

    for features in probe_input.features.values():
        assert features.shape == (4, model.cfg.n_heads * model.cfg.d_head)


def test_per_location_pooling_types(model):
    config = _config(
        pool={ModelLocation.POSITION: PoolingType.MEAN, ModelLocation.HEAD: PoolingType.MAX},
        activation=ModelLayerLocation.ATTENTION_Z,
    )
    probe_input = ProbeInput(config, POSITIVE, NEGATIVE).prepare(model)

    for features in probe_input.features.values():
        assert features.shape == (4, model.cfg.d_head)

    tokens = model.to_tokens(POSITIVE[0])
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    expected = cache["blocks.0.attn.hook_z"][0].mean(dim=0).amax(dim=0).to("cpu", torch.float32)
    layer_output = next(single.output for single in probe_input.index if single.source.layer() == 0)
    assert torch.allclose(probe_input.features[layer_output][0], expected, atol=1e-5)


def test_position_grid_with_unequal_length_prompts_raises(model):
    config = _config(location=[ModelLocation.LAYER, ModelLocation.POSITION])
    with pytest.raises(ValueError, match="prompt length varies"):
        ProbeInput(config, POSITIVE, NEGATIVE).prepare(model)


def _position_filter_config(*positions: int) -> ProbeConfig:
    return ProbeConfig(
        type=ProbeType.LINEAR,
        pool={},
        metric=[ProbeMetric.ACCURACY],
        source=ModelLocatorConfig(
            activation=ModelLayerLocation.RESIDUAL_POST,
            location=[ModelLocation.LAYER, ModelLocation.POSITION],
            filter=Filter.from_config([{"position": position} for position in positions]),
        ),
    )


def test_last_position_with_equal_length_prompts(model):
    probe_input = ProbeInput(_position_filter_config(-1), EQUAL_POSITIVE, EQUAL_NEGATIVE).prepare(model)

    assert probe_input.index.axis_range == [model.cfg.n_layers, 1]
    for features in probe_input.features.values():
        assert features.shape == (2, model.cfg.d_model)


def test_last_position_with_unequal_length_prompts_uses_each_prompts_last_token(model):
    probe_input = ProbeInput(_position_filter_config(-1), POSITIVE, NEGATIVE).prepare(model)

    assert probe_input.index.axis_range == [model.cfg.n_layers, 1]
    for features in probe_input.features.values():
        assert features.shape == (4, model.cfg.d_model)

    layer_output = next(s.output for s in probe_input.index if s.source.layer() == 0)
    for example, prompt in enumerate(POSITIVE + NEGATIVE):
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        expected = cache["blocks.0.hook_resid_post"][0, -1].to("cpu", torch.float32)
        assert torch.allclose(probe_input.features[layer_output][example], expected, atol=1e-5)


@pytest.mark.parametrize("position", [40, -40])
def test_out_of_range_position_raises(model, position):
    with pytest.raises(ValueError, match="out of range for the shortest prompt"):
        ProbeInput(_position_filter_config(position), POSITIVE, NEGATIVE).prepare(model)


def test_mixed_anchor_positions_with_unequal_length_prompts(model):
    probe_input = ProbeInput(_position_filter_config(0, -1), POSITIVE, NEGATIVE).prepare(model)

    assert probe_input.index.axis_range == [model.cfg.n_layers, 2]
    assert {single.source[ModelLocation.POSITION] for single in probe_input.index} == {0, -1}


def test_mixed_absolute_and_relative_positions_with_unequal_length_prompts(model):
    # 2 (from the start) and -2 (from the end) are two distinct grid points even though they collide
    # on a 4-token prompt; each is sliced against its own prompt.
    probe_input = ProbeInput(_position_filter_config(2, -2), POSITIVE, NEGATIVE).prepare(model)

    assert probe_input.index.axis_range == [model.cfg.n_layers, 2]
    assert {single.source[ModelLocation.POSITION] for single in probe_input.index} == {2, -2}

    layer_output = next(s.output for s in probe_input.index if s.source.layer() == 0)
    assert probe_input.features[layer_output].shape == (4, model.cfg.d_model)


def test_position_grid_with_equal_length_prompts(model):
    config = _config(location=[ModelLocation.LAYER, ModelLocation.POSITION])
    probe_input = ProbeInput(config, EQUAL_POSITIVE, EQUAL_NEGATIVE).prepare(model)

    positions = model.to_tokens(EQUAL_POSITIVE[0]).shape[1]
    assert probe_input.index.axis_range == [model.cfg.n_layers, positions]
    for features in probe_input.features.values():
        assert features.shape == (2, model.cfg.d_model)


def test_pattern_activation_with_unequal_length_prompts_raises(model):
    config = _config(
        pool={}, activation=ModelLayerLocation.ATTENTION_PATTERN, location=[ModelLocation.LAYER, ModelLocation.HEAD]
    )
    with pytest.raises(ValueError, match="Feature size varies"):
        ProbeInput(config, POSITIVE, NEGATIVE).prepare(model)


def test_pattern_position_pooling_spans_both_position_axes(model):
    config = _config(
        activation=ModelLayerLocation.ATTENTION_PATTERN, location=[ModelLocation.LAYER, ModelLocation.HEAD]
    )
    probe_input = ProbeInput(config, POSITIVE, NEGATIVE).prepare(model)

    for features in probe_input.features.values():
        assert features.shape == (4, 1)


def test_empty_prompt_lists_raise(model):
    with pytest.raises(ValueError, match="positive and negative prompts"):
        ProbeInput(_config(), POSITIVE, []).prepare(model)
    with pytest.raises(ValueError, match="positive and negative prompts"):
        ProbeInput(_config(), [], NEGATIVE).prepare(model)
