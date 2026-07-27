import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
from transformer_lens import HookedTransformer

from intex.model.cache import CachedModel
from intex.model.config import Filter, Group, Incremental, ModelLayerLocation, ModelLocation
from intex.operation.config import ConfigFactory
from intex.patching.collector import PatchingCollector
from intex.patching.config import (
    PatchingConfig,
    PatchingElementConfig,
    PatchingMetric,
    PatchingType,
    SinglePatchingConfig,
)
from intex.patching.input import PatchingInput
from intex.patching.patcher import Patcher

from ..conftest import RenderedFigure, forbid_model_runs, rendered_figure

SOURCE_PROMPT = "The capital of France is"
TARGET_PROMPT = "The capital of Spain is"


def _metric(collector: PatchingCollector) -> torch.Tensor:
    """The result frame's metric column, in index order — one value per patched run."""
    return torch.tensor(collector.result[collector.operation_input.config.metric.value].to_numpy(), dtype=torch.float32)


def _run(model: HookedTransformer, config: PatchingConfig) -> torch.Tensor:
    collector = Patcher(model).apply(
        PatchingInput(config, SOURCE_PROMPT, TARGET_PROMPT), return_collector=True, verbose=False
    )
    assert collector is not None
    return _metric(collector)


def _generations(model: HookedTransformer) -> tuple[int, int]:
    """The tokens the two prompts predict — the pair every logit-diff metric compares."""
    with torch.no_grad():
        return int(model(SOURCE_PROMPT)[0, -1].argmax()), int(model(TARGET_PROMPT)[0, -1].argmax())


def _source_activation(model: HookedTransformer, name: str) -> torch.Tensor:
    with torch.no_grad():
        _, cache = model.run_with_cache(model.to_tokens(SOURCE_PROMPT), names_filter=[name])
    return cache[name]


def _expected_logit_diff(model: HookedTransformer, fwd_hooks: list) -> torch.Tensor:
    """The metric a patched run must produce, from running the same patch by hand.

    The hooks say what the patch writes and where; the logit diff is then read straight off the
    patched run, so nothing about the expected value is frozen into the test.
    """
    source_token, target_token = _generations(model)
    with torch.no_grad():
        patched = model.run_with_hooks(model.to_tokens(TARGET_PROMPT), fwd_hooks=fwd_hooks)
    return patched[0, -1, source_token] - patched[0, -1, target_token]


def _last_layer_resid_element() -> PatchingElementConfig:
    return PatchingElementConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelLocation.LAYER],
        filter=Filter(inclusion_terms=[{ModelLocation.LAYER: -1}]),
        type=PatchingType.REPLACE,
    )


def _last_layer_resid_patch_config() -> PatchingConfig:
    """Replace the last layer's full residual stream (all positions) source -> target."""
    element = _last_layer_resid_element()
    return PatchingConfig(
        config_id="last_layer_resid_patch",
        metric=PatchingMetric.NORMALIZED_LOGIT_DIFF,
        patch=[SinglePatchingConfig(source=element, target=element)],
    )


def _last_layer_resid_zero_ablation_config() -> PatchingConfig:
    """Zero-ablate the last layer's full residual stream (all positions): set every value to 0."""
    source = PatchingElementConfig(type=PatchingType.VALUE_REPLACE)  # produces zeros
    target = _last_layer_resid_element()  # writes them into the activation
    return PatchingConfig(
        config_id="last_layer_resid_zero_ablation",
        metric=PatchingMetric.LOGIT_DIFF,
        patch=[SinglePatchingConfig(source=source, target=target)],
    )


def _resid_value_scale_config(scale: float, layer: int) -> PatchingConfig:
    """Scale a layer's source residual stream (all positions) by ``scale``, then write it into the target.

    The scaled layer is intentionally not the final one: ``resid.post`` of the last layer feeds straight
    into ``ln_final``, whose normalization cancels any uniform scaling, so the scale would have no effect
    on the logits. At an intermediate layer the scaled residual is carried forward through the skip
    connections, so the scale genuinely changes the output.
    """
    source = PatchingElementConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelLocation.LAYER],
        filter=Filter(inclusion_terms=[{ModelLocation.LAYER: layer}]),
        type=PatchingType.VALUE_SCALE,
        value=scale,
    )
    target = PatchingElementConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelLocation.LAYER],
        filter=Filter(inclusion_terms=[{ModelLocation.LAYER: layer}]),
        type=PatchingType.REPLACE,
    )
    return PatchingConfig(
        config_id="resid_value_scale",
        metric=PatchingMetric.LOGIT_DIFF,
        patch=[SinglePatchingConfig(source=source, target=target)],
    )


def _mlp_mhsa_pattern_elements(layer: int, head: int) -> list[PatchingElementConfig]:
    """Three components of one layer, each replaced from the source run:

    - the MLP post-activation neurons (``mlp.post``),
    - the multi-head self-attention queries (``attn.q``), and
    - a single attention head's pattern (``attn.pattern`` filtered to ``head``).
    """
    mlp = PatchingElementConfig(
        activation=ModelLayerLocation.MLP_POST,
        location=[ModelLocation.LAYER],
        filter=Filter(inclusion_terms=[{ModelLocation.LAYER: layer}]),
        type=PatchingType.REPLACE,
    )
    mhsa = PatchingElementConfig(
        activation=ModelLayerLocation.ATTENTION_Q,
        location=[ModelLocation.LAYER],
        filter=Filter(inclusion_terms=[{ModelLocation.LAYER: layer}]),
        type=PatchingType.REPLACE,
    )
    pattern = PatchingElementConfig(
        activation=ModelLayerLocation.ATTENTION_PATTERN,
        location=[ModelLocation.LAYER, ModelLocation.HEAD],
        filter=Filter(inclusion_terms=[{ModelLocation.LAYER: layer, ModelLocation.HEAD: head}]),
        type=PatchingType.REPLACE,
    )
    return [mlp, mhsa, pattern]


def _patch_config(*elements: PatchingElementConfig, config_id: str = "patch") -> PatchingConfig:
    """Patches sharing one config are applied together, on a single forward pass."""
    return PatchingConfig(
        config_id=config_id,
        metric=PatchingMetric.LOGIT_DIFF,
        patch=[SinglePatchingConfig(source=element, target=element) for element in elements],
    )


def test_activation_patching_from_source_into_target(model: HookedTransformer) -> None:
    torch.manual_seed(42)

    # A full residual replacement requires matching sequence lengths between the two prompts.
    assert model.to_tokens(SOURCE_PROMPT).shape == model.to_tokens(TARGET_PROMPT).shape

    patching_input = PatchingInput(_last_layer_resid_patch_config(), SOURCE_PROMPT, TARGET_PROMPT)

    collector = Patcher(model).apply(patching_input, return_collector=True, verbose=False)

    # One patch (the single last layer) -> a one-row result.
    assert collector is not None
    result = _metric(collector)
    assert result.shape == (1,)

    assert torch.allclose(result.squeeze(), torch.tensor(1.0))


def test_caches_restricted_to_used_activations(model: HookedTransformer) -> None:
    patching_input = PatchingInput(_last_layer_resid_patch_config(), SOURCE_PROMPT, TARGET_PROMPT).prepare(model)

    # Only the last layer's residual is read; both caches are filtered down to it.
    expected = set().union(*(idx.source_activation_names() for idx in patching_input.index))
    assert expected == {f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"}
    # A bridge fires the block's own output hook under a second name for the same activation; once
    # that alias is resolved, each cache holds exactly the one activation the index named.
    alias = {f"blocks.{model.cfg.n_layers - 1}.hook_out"}
    assert set(patching_input.source_cache.keys()) - alias == expected
    assert set(patching_input.target_cache.keys()) - alias == expected

    _, full_cache = model.run_with_cache(patching_input.source_tokens)
    assert set(patching_input.source_cache.keys()) < set(full_cache.keys())


def test_zero_ablation_patching(model: HookedTransformer) -> None:
    torch.manual_seed(42)

    # Zero ablation writes zeros into the target run, so the source run's activations are never read;
    # its forward pass only supplies the source generation token that the logit-diff metric compares
    # against the target one. Distinct prompts keep those two tokens distinct.
    patching_input = PatchingInput(_last_layer_resid_zero_ablation_config(), SOURCE_PROMPT, TARGET_PROMPT)

    collector = Patcher(model).apply(patching_input, return_collector=True, verbose=False)

    # One patch (the single last layer) -> a one-row result.
    assert collector is not None
    result = _metric(collector)
    assert result.shape == (1,)

    # The same ablation done by hand: the last layer's residual is zeroed on the target run.
    name = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"
    expected = _expected_logit_diff(model, [(name, lambda activation, hook: torch.zeros_like(activation))])
    assert torch.allclose(result.squeeze(), expected, atol=1e-5)


# Intermediate layer whose residual stream is scaled; see ``_resid_value_scale_config`` for why it is
# not the final layer.
_VALUE_SCALE_LAYER = 6


def _run_value_scale(model: HookedTransformer, scale: float) -> torch.Tensor:
    # Scaling the source residual and writing it into the target requires matching sequence lengths.
    assert model.to_tokens(SOURCE_PROMPT).shape == model.to_tokens(TARGET_PROMPT).shape

    patching_input = PatchingInput(_resid_value_scale_config(scale, _VALUE_SCALE_LAYER), SOURCE_PROMPT, TARGET_PROMPT)

    collector = Patcher(model).apply(patching_input, return_collector=True, verbose=False)

    # One patch (a single layer) -> a one-row result.
    assert collector is not None
    result = _metric(collector)
    assert result.shape == (1,)
    return result.squeeze()


def test_value_scale_patching(model: HookedTransformer) -> None:
    torch.manual_seed(42)

    scaled = _run_value_scale(model, scale=2.0)
    unscaled = _run_value_scale(model, scale=1.0)

    # Each run is checked against the same scaling done by hand: the source's residual, multiplied,
    # written over the target's at that layer.
    name = f"blocks.{_VALUE_SCALE_LAYER}.hook_resid_post"
    source_residual = _source_activation(model, name)
    for result, scale in ((scaled, 2.0), (unscaled, 1.0)):
        expected = _expected_logit_diff(model, [(name, lambda activation, hook, s=scale: s * source_residual)])
        assert torch.allclose(result, expected, atol=1e-5)

    # Scaling by 2x is not the same as writing the residual through unchanged, so the scale takes effect.
    assert not torch.allclose(scaled, unscaled)


# Layer and head whose MLP, attention output, and head pattern are patched together below.
_MULTI_PATCH_LAYER = 5
_MULTI_PATCH_HEAD = 7


def _multi_patch_hooks(model: HookedTransformer) -> list:
    """The three replacements done by hand, from the source run's own activations."""
    layer, head = _MULTI_PATCH_LAYER, _MULTI_PATCH_HEAD
    post, queries, pattern = (
        f"blocks.{layer}.mlp.hook_post",
        f"blocks.{layer}.attn.hook_q",
        f"blocks.{layer}.attn.hook_pattern",
    )
    # Read from the source run up front: a forward pass inside a hook would run the hooks again.
    source_post, source_queries, source_pattern = (_source_activation(model, name) for name in (post, queries, pattern))

    def patch_one_head(activation: torch.Tensor, hook) -> torch.Tensor:
        patched = activation.clone()
        patched[:, head] = source_pattern[:, head]
        return patched

    return [
        (post, lambda activation, hook: source_post),
        (queries, lambda activation, hook: source_queries),
        (pattern, patch_one_head),
    ]


def test_multi_config_patching_mlp_mhsa_and_pattern(model: HookedTransformer) -> None:
    torch.manual_seed(42)

    # Replacing full activations (and a head's pattern) from source requires matching sequence lengths.
    assert model.to_tokens(SOURCE_PROMPT).shape == model.to_tokens(TARGET_PROMPT).shape

    elements = _mlp_mhsa_pattern_elements(_MULTI_PATCH_LAYER, _MULTI_PATCH_HEAD)
    patching_input = PatchingInput(_patch_config(*elements), SOURCE_PROMPT, TARGET_PROMPT)

    collector = Patcher(model).apply(patching_input, return_collector=True, verbose=False)

    # The config carries three patches; each becomes its own index, targeting -- in order -- the MLP
    # post-activation, the MHSA queries, and the attention pattern.
    assert [idx.target_config.activation for idx in patching_input.index] == [
        ModelLayerLocation.MLP_POST,
        ModelLayerLocation.ATTENTION_Q,
        ModelLayerLocation.ATTENTION_PATTERN,
    ]

    # All three patches are applied together on a single forward pass and the collector keys the result
    # off the first patch's (single) output coordinate -> a one-row result.
    assert collector is not None
    result = _metric(collector)
    assert result.shape == (1,)

    # The three replacements done by hand, on one forward pass, give the same logit diff...
    assert torch.allclose(result.squeeze(), _expected_logit_diff(model, _multi_patch_hooks(model)), atol=1e-5)

    # ...and none of the three on its own does, so all of them are taking effect at once.
    for element in elements:
        alone = _run(model, _patch_config(element, config_id="single_patch"))
        assert not torch.allclose(result.squeeze(), alone.squeeze(), atol=1e-3)


def test_the_kl_metric_scores_the_patched_distribution(model: HookedTransformer) -> None:
    # Position 0 is the prompts' shared prefix, so writing the source's activation there is a no-op:
    # the patched run is the target run, which makes the expected divergence computable from two
    # plain forward passes. Note what the metric compares — the source's distribution at every
    # position against the target's at every position, then against the patched *last* position.
    element = PatchingElementConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        coordinates=[{ModelLocation.LAYER: -1, ModelLocation.POSITION: 0}],
        type=PatchingType.REPLACE,
    )
    config = PatchingConfig(
        config_id="kl_no_op",
        metric=PatchingMetric.KL_DIFF,
        patch=[SinglePatchingConfig(source=element, target=element)],
    )

    result = _run(model, config)

    with torch.no_grad():
        source_probabilities = model(SOURCE_PROMPT).softmax(dim=-1)
        target_logits = model(TARGET_PROMPT)
    divergence = torch.nn.functional.kl_div
    expected = divergence(
        source_probabilities.log(), target_logits.softmax(dim=-1), reduction="batchmean"
    ) - divergence(source_probabilities.log(), target_logits[0, -1].softmax(dim=-1), reduction="batchmean")
    assert torch.allclose(result.squeeze(), expected, atol=1e-4)


def _explicit_element(
    coordinates: list[dict], activation: ModelLayerLocation | None = ModelLayerLocation.RESIDUAL_POST
) -> PatchingElementConfig:
    return PatchingElementConfig(
        activation=activation,
        coordinates=coordinates,
        type=PatchingType.REPLACE,
    )


def _explicit_patch_config(
    source_coordinates: list[dict],
    target_coordinates: list[dict],
    activation: ModelLayerLocation | None = ModelLayerLocation.RESIDUAL_POST,
) -> PatchingConfig:
    """Patch the exact coordinates listed: source coordinate i is read from the source run and written at
    target coordinate i of the target run (one patched run per pair)."""
    return PatchingConfig(
        config_id="explicit_patch",
        metric=PatchingMetric.NORMALIZED_LOGIT_DIFF,
        patch=[
            SinglePatchingConfig(
                source=_explicit_element(source_coordinates, activation),
                target=_explicit_element(target_coordinates, activation),
            )
        ],
    )


def _run_explicit(
    model: HookedTransformer,
    source_coordinates: list[dict],
    target_coordinates: list[dict],
    save_dir: str | None = None,
    activation: ModelLayerLocation | None = ModelLayerLocation.RESIDUAL_POST,
) -> torch.Tensor:
    patching_input = PatchingInput(
        _explicit_patch_config(source_coordinates, target_coordinates, activation), SOURCE_PROMPT, TARGET_PROMPT
    )
    collector = Patcher(model).apply(patching_input, save_dir=save_dir, return_collector=True, verbose=False)
    assert collector is not None
    return _metric(collector)


def test_explicit_coordinates_matches_declarative_patch(model: HookedTransformer) -> None:
    torch.manual_seed(42)

    # The explicit twin of ``test_activation_patching_from_source_into_target``: one coordinate covering
    # the last layer's full residual stream reproduces the declarative filter's result.
    coordinate = [{ModelLocation.LAYER: -1}]
    result = _run_explicit(model, coordinate, coordinate)

    assert result.shape == (1,)
    assert torch.allclose(result.squeeze(), torch.tensor(1.0))


def test_explicit_position_is_respected(model: HookedTransformer) -> None:
    torch.manual_seed(42)

    # The final token's logits depend only on the last position of the final layer's residual, so
    # patching just that coordinate reproduces the full source run (metric 1.0). The prompts share their
    # prefix, so position 0 carries identical activations in both runs and patching it is a no-op
    # (metric 0.0). If the position were ignored (patching every position), both would give 1.0.
    last_layer_last_position = [{ModelLocation.LAYER: -1, ModelLocation.POSITION: -1}]
    last_layer_first_position = [{ModelLocation.LAYER: -1, ModelLocation.POSITION: 0}]
    last_position = _run_explicit(model, last_layer_last_position, last_layer_last_position)
    first_position = _run_explicit(model, last_layer_first_position, last_layer_first_position)

    assert torch.allclose(last_position.squeeze(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(first_position.squeeze(), torch.tensor(0.0), atol=1e-5)


def test_explicit_pairing_and_artifacts(model: HookedTransformer, tmp_path: Path) -> None:
    torch.manual_seed(42)

    # Two pairs reading different source layers into different target positions of layer 7.
    source_coordinates = [
        {ModelLocation.LAYER: 5, ModelLocation.POSITION: -1},
        {ModelLocation.LAYER: 6, ModelLocation.POSITION: -1},
    ]
    target_coordinates = [
        {ModelLocation.LAYER: 7, ModelLocation.POSITION: -1},
        {ModelLocation.LAYER: 7, ModelLocation.POSITION: -2},
    ]
    save_dir = tmp_path / "explicit_patch_run"
    paired = _run_explicit(model, source_coordinates, target_coordinates, save_dir=str(save_dir))

    # Explicit coordinates need not form a grid, so the result is flat: one entry per pair, in order,
    # each matching the same pair patched on its own.
    assert paired.shape == (2,)
    first = _run_explicit(model, source_coordinates[:1], target_coordinates[:1])
    second = _run_explicit(model, source_coordinates[1:], target_coordinates[1:])
    assert torch.allclose(paired, torch.cat([first, second]))
    assert not torch.allclose(first, second)

    # index-0.csv pairs the source/target coordinates with the flat output axis.
    index_df = pd.read_csv(save_dir / "index-0.csv")
    assert len(index_df) == 2
    assert any(col.startswith("source_") for col in index_df.columns)
    assert any(col.startswith("target_") for col in index_df.columns)
    assert [col for col in index_df.columns if col.startswith("index_")] == ["index_coordinate"]
    assert index_df["index_coordinate"].to_list() == [0, 1]


def test_explicit_per_coordinate_activation(model: HookedTransformer) -> None:
    torch.manual_seed(42)

    # Each coordinate carries its own activation: pair 0 patches the final layer's residual at the last
    # position, reproducing the full source run (metric exactly 1.0, as established above); pair 1
    # patches mlp.out at the same coordinate, which only swaps the MLP component's contribution, so its
    # metric lands well away from 1.0 -- proving each row is patched at its own activation's hook.
    coordinates = [
        {
            ModelLocation.ACTIVATION: ModelLayerLocation.RESIDUAL_POST,
            ModelLocation.LAYER: -1,
            ModelLocation.POSITION: -1,
        },
        {ModelLocation.ACTIVATION: ModelLayerLocation.MLP_OUT, ModelLocation.LAYER: -1, ModelLocation.POSITION: -1},
    ]
    result = _run_explicit(model, coordinates, coordinates, activation=None)

    assert result.shape == (2,)
    assert torch.allclose(result[0], torch.tensor(1.0), atol=1e-5)
    assert not torch.allclose(result[1], torch.tensor(1.0), atol=1e-2)


def test_two_sources_differing_only_in_activation_keep_their_own_metric(model: HookedTransformer) -> None:
    torch.manual_seed(42)

    # Explicit source coordinates against a declarative target: that mix is what sends the index
    # through the factorized output branch, which drops the activation column. Both coordinates sit at
    # the same layer/position/neuron, so activation is the only thing telling the two runs apart --
    # and a collector keyed on that coordinate would file both under one cell, reporting whichever ran
    # last on both rows.
    source = PatchingElementConfig(
        type=PatchingType.REPLACE,
        coordinates=[
            {
                ModelLocation.ACTIVATION: activation,
                ModelLocation.LAYER: 0,
                ModelLocation.POSITION: -1,
                ModelLocation.NEURON: 1,
            }
            for activation in (ModelLayerLocation.MLP_POST, ModelLayerLocation.MLP_PRE)
        ],
    )
    # One target row, so ``zip_fill`` holds it steady while the two source coordinates vary.
    target = PatchingElementConfig(
        type=PatchingType.REPLACE,
        activation=ModelLayerLocation.MLP_POST,
        location=[ModelLocation.LAYER],
        filter=Filter(inclusion_terms=[{ModelLocation.LAYER: 0}]),
    )
    config = PatchingConfig(
        config_id="same_coordinate_two_activations",
        metric=PatchingMetric.LOGIT_DIFF,
        patch=[SinglePatchingConfig(source=source, target=target)],
    )
    patching_input = PatchingInput(config, SOURCE_PROMPT, TARGET_PROMPT)

    collector = Patcher(model).apply(patching_input, return_collector=True, verbose=False)
    assert collector is not None
    result = collector.result

    # The two coordinates are one grid point once the activation is dropped...
    assert patching_input.index[0].output_df.drop_duplicates().shape[0] == 1
    # ...yet each run keeps its own row, naming the activation it read and the metric it produced.
    assert result[f"source_{ModelLocation.ACTIVATION.value}"].to_list() == [
        ModelLayerLocation.MLP_POST,
        ModelLayerLocation.MLP_PRE,
    ]
    assert _metric(collector)[0] != _metric(collector)[1]


def test_artifacts_saved_to_disk(model: HookedTransformer, tmp_path: Path) -> None:
    torch.manual_seed(42)

    # Build the config through ConfigFactory by persisting it to a YAML file and loading it back,
    # mirroring how configs are sourced in production rather than constructing PatchingConfig directly.
    element = {
        "activation": "resid.post",
        "location": ["layer"],
        "filter": [{"layer": -1}],
        "type": "replace",
    }
    config_yaml = tmp_path / "configs.yaml"
    config_yaml.write_text(
        yaml.safe_dump(
            {"last_layer_resid_patch": {"metric": "logit_diff", "patch": [{"source": element, "target": element}]}}
        )
    )
    config = ConfigFactory[PatchingConfig](str(config_yaml)).create("last_layer_resid_patch")
    patching_input = PatchingInput(config, SOURCE_PROMPT, TARGET_PROMPT)

    save_dir = tmp_path / "patch_run"
    collector = Patcher(model).apply(patching_input, save_dir=str(save_dir), return_collector=True, verbose=False)
    assert collector is not None

    # The three artifacts the collector is expected to persist.
    result_path = save_dir / "result.csv"
    index_path = save_dir / "index-0.csv"
    config_path = save_dir / "config.json"
    assert result_path.exists()
    assert index_path.exists()
    assert config_path.exists()
    assert not (save_dir / "result.pt").exists()

    # result.csv round-trips the in-memory frame: the source/target/index columns of the patched
    # element, plus one column per metric, named after it.
    result_df = pd.read_csv(result_path)
    assert len(result_df) == len(collector.result) == 1
    assert result_df["logit_diff"].to_list() == pytest.approx(collector.result["logit_diff"].to_list())
    assert any(col.startswith("source_") for col in result_df.columns)
    assert any(col.startswith("target_") for col in result_df.columns)
    assert any(col.startswith("index_") for col in result_df.columns)

    # config.json mirrors the patching config that produced the run.
    saved_config = json.loads(config_path.read_text())
    assert saved_config["config_id"] == "last_layer_resid_patch"
    assert saved_config["metric"] == "logit_diff"
    assert len(saved_config["patch"]) == 1
    assert saved_config["patch"][0]["source"]["type"] == "replace"
    assert saved_config["patch"][0]["target"]["type"] == "replace"

    # index-0.csv carries one row per result entry, with the source/target/index columns.
    index_df = pd.read_csv(index_path)
    assert len(index_df) == len(collector.result)
    assert any(col.startswith("source_") for col in index_df.columns)
    assert any(col.startswith("target_") for col in index_df.columns)
    assert any(col.startswith("index_") for col in index_df.columns)


def _gaussian_noise_config() -> PatchingConfig:
    """Scale the target's own residual by seeded noise, then write it back over itself."""
    source = PatchingElementConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelLocation.LAYER],
        filter=Filter(inclusion_terms=[{ModelLocation.LAYER: _VALUE_SCALE_LAYER}]),
        type=PatchingType.GAUSSIAN_NOISE,
    )
    target = PatchingElementConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelLocation.LAYER],
        filter=Filter(inclusion_terms=[{ModelLocation.LAYER: _VALUE_SCALE_LAYER}]),
        type=PatchingType.REPLACE,
    )
    return PatchingConfig(
        config_id="gaussian_noise",
        metric=PatchingMetric.LOGIT_DIFF,
        patch=[SinglePatchingConfig(source=source, target=target)],
    )


def test_gaussian_noise_patching(model: HookedTransformer) -> None:
    torch.manual_seed(42)

    noised = _run(model, _gaussian_noise_config())

    # The strategy draws from a seeded generator, so the run repeats exactly...
    assert noised.shape == (1,)
    assert torch.isfinite(noised).all()
    assert torch.equal(_run(model, _gaussian_noise_config()), noised)
    # ...and noising the activation is not the same as replacing it with the source's.
    assert not torch.allclose(noised, _run(model, _resid_value_scale_config(1.0, _VALUE_SCALE_LAYER)))


@pytest.mark.parametrize(
    ("element", "message"),
    [
        ("source", "Unknown source strategy"),
        ("target", "Unknown target strategy"),
    ],
)
def test_a_patching_type_that_is_not_known_raises(model: HookedTransformer, element: str, message: str) -> None:
    # A config built in code can name a type no strategy implements — and 'value_replace' is one only
    # a source may carry, so a target is rejected the same way.
    source, target = _last_layer_resid_element(), _last_layer_resid_element()
    (source if element == "source" else target).type = "bogus" if element == "source" else "value_replace"
    config = PatchingConfig(
        config_id="unknown_strategy",
        metric=PatchingMetric.LOGIT_DIFF,
        patch=[SinglePatchingConfig(source=source, target=target)],
    )

    with pytest.raises(ValueError, match=message):
        Patcher(model).apply(PatchingInput(config, SOURCE_PROMPT, TARGET_PROMPT), verbose=False)


def test_a_patching_metric_that_is_not_known_raises(model: HookedTransformer) -> None:
    config = _last_layer_resid_patch_config()
    config.metric = "accuracy"

    with pytest.raises(ValueError, match="Unknown patching metric"):
        Patcher(model).apply(PatchingInput(config, SOURCE_PROMPT, TARGET_PROMPT), verbose=False)


def test_a_prepared_input_moves_to_a_device(model: HookedTransformer) -> None:
    patching_input = PatchingInput(_last_layer_resid_patch_config(), SOURCE_PROMPT, TARGET_PROMPT).prepare(model)
    # The baselines and probabilities are built on first read, so ask for them before moving.
    for lazy in (patching_input.source_baseline, patching_input.target_baseline):
        assert lazy is not None
    for lazy in (patching_input.source_probabilities, patching_input.target_probabilities):
        assert lazy is not None

    moved = patching_input.to("cpu")

    assert moved is patching_input
    tensors = [
        moved.source_tokens,
        moved.target_tokens,
        moved.source_logits,
        moved.target_logits,
        moved.source_baseline,
        moved.target_baseline,
        moved.source_probabilities,
        moved.target_probabilities,
    ]
    assert all(tensor.device.type == "cpu" for tensor in tensors)
    assert all(activation.device.type == "cpu" for activation in moved.source_cache.values())
    assert all(activation.device.type == "cpu" for activation in moved.target_cache.values())


def test_a_run_without_a_collector_still_writes_its_artifacts(model: HookedTransformer, tmp_path: Path) -> None:
    torch.manual_seed(42)

    # Not asking for the collector is the default, and how the example scripts run an operation.
    collector = Patcher(model).apply(
        PatchingInput(_last_layer_resid_patch_config(), SOURCE_PROMPT, TARGET_PROMPT),
        return_collector=True,
        verbose=False,
    )
    assert collector is not None

    save_dir = tmp_path / "patch_run"
    returned = Patcher(model).apply(
        PatchingInput(_last_layer_resid_patch_config(), SOURCE_PROMPT, TARGET_PROMPT),
        save_dir=str(save_dir),
        verbose=False,
    )

    # Nothing comes back, but the run did all of its work and left it on disk.
    assert returned is None
    pd.testing.assert_frame_equal(pd.read_csv(save_dir / "result.csv"), collector.result)
    assert (save_dir / "index-0.csv").exists()


# --------------------------------------------------------------------------------------------------
# Configuration — patching configs come from YAML, where a source and a target often alias one anchor
# --------------------------------------------------------------------------------------------------
ELEMENT_YAML = {"activation": "resid.post", "location": ["layer"], "filter": [{"layer": -1}], "type": "replace"}


def _patch_yaml(**overrides) -> dict:
    return {"metric": "logit_diff", "patch": [{"source": ELEMENT_YAML, "target": ELEMENT_YAML}], **overrides}


def test_a_yaml_config_runs_the_same_way_a_code_built_one_does(model: HookedTransformer, config_from_yaml) -> None:
    from_file = config_from_yaml(PatchingConfig, "last_layer_resid_patch", _patch_yaml())
    element = _last_layer_resid_element()
    in_code = PatchingConfig(
        metric=PatchingMetric.LOGIT_DIFF, patch=[SinglePatchingConfig(source=element, target=element)]
    )

    assert from_file.metric is PatchingMetric.LOGIT_DIFF
    assert from_file.patch[0].source.activation is ModelLayerLocation.RESIDUAL_POST
    assert from_file.patch[0].source.location == [ModelLocation.LAYER]
    assert from_file.patch[0].source.type is PatchingType.REPLACE

    def run(config: PatchingConfig) -> torch.Tensor:
        patching_input = PatchingInput(config, SOURCE_PROMPT, TARGET_PROMPT)
        collector = Patcher(model).apply(patching_input, return_collector=True, verbose=False)
        assert collector is not None
        return _metric(collector)

    assert torch.equal(run(from_file), run(in_code))


def test_a_source_and_a_target_may_alias_the_same_yaml_anchor(model: HookedTransformer, tmp_path: Path) -> None:
    # A YAML anchor gives the deserializer the very same mapping twice; it must not consume it.
    config_yaml = tmp_path / "configs.yaml"
    config_yaml.write_text(
        "anchored:\n"
        "  metric: logit_diff\n"
        "  patch:\n"
        "    - source: &element\n"
        "        activation: resid.post\n"
        "        location: [layer]\n"
        "        filter: [{layer: -1}]\n"
        "        type: replace\n"
        "      target: *element\n"
    )
    config = ConfigFactory[PatchingConfig](str(config_yaml)).create("anchored")

    assert config.patch[0].source == config.patch[0].target
    collector = Patcher(model).apply(
        PatchingInput(config, SOURCE_PROMPT, TARGET_PROMPT), return_collector=True, verbose=False
    )
    assert collector is not None
    assert len(collector.result) == 1


def test_the_omitted_keys_of_an_element_fall_back_to_their_defaults(model: HookedTransformer, config_from_yaml) -> None:
    # A source that names nothing but its type produces zeros — the zero-ablation shape, from YAML.
    raw = {
        "metric": "logit_diff",
        "patch": [{"source": {"type": "value_replace"}, "target": ELEMENT_YAML}],
    }
    config = config_from_yaml(PatchingConfig, "zero_ablation", raw)
    source = config.patch[0].source

    assert source.activation is None
    assert source.location == []
    assert (source.filter, source.group, source.incremental) == (Filter(), Group(), Incremental())
    assert source.value == 0.0

    # Those defaults are the zero ablation built in code, so the two runs agree.
    assert torch.equal(_run(model, config), _run(model, _last_layer_resid_zero_ablation_config()))


def test_an_exclusion_term_is_deserialized_into_its_own_filter(model: HookedTransformer, config_from_yaml) -> None:
    element = {**ELEMENT_YAML, "filter": [{"layer": {"range": [0, 2]}}, {"not": {"layer": 1}}]}
    config = config_from_yaml(
        PatchingConfig, "filtered", {"metric": "logit_diff", "patch": [{"source": element, "target": element}]}
    )

    assert config.patch[0].source.filter == Filter(
        inclusion_terms=[{ModelLocation.LAYER: {"range": [0, 2]}}], exclusion_terms=[{ModelLocation.LAYER: 1}]
    )

    collector = Patcher(model).apply(
        PatchingInput(config, SOURCE_PROMPT, TARGET_PROMPT), return_collector=True, verbose=False
    )
    assert collector is not None
    assert len(collector.result) == 2  # layers 0 and 2 survive the exclusion


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (_patch_yaml(typ="replace"), "Unknown keys for PatchingConfig.*typ"),
        (
            {"metric": "logit_diff", "patch": [{"source": {"activation": "resid.post"}, "target": ELEMENT_YAML}]},
            "Missing required keys for PatchingElementConfig.*type",
        ),
        (
            {"metric": "logit_diff", "patch": [{"source": "replace", "target": ELEMENT_YAML}]},
            "Expected a mapping for PatchingElementConfig",
        ),
        ({"patch": []}, "Missing required keys for PatchingConfig.*metric"),
        (_patch_yaml(metric="accuracy"), "accuracy"),
    ],
)
def test_a_malformed_patching_config_is_rejected_when_it_is_loaded(config_from_yaml, raw, message) -> None:
    with pytest.raises(ValueError, match=message):
        config_from_yaml(PatchingConfig, "broken", raw)


def test_a_config_id_the_file_does_not_have_raises(tmp_path: Path) -> None:
    config_yaml = tmp_path / "configs.yaml"
    config_yaml.write_text(yaml.safe_dump({"my_patch": _patch_yaml()}))
    factory = ConfigFactory[PatchingConfig](str(config_yaml))

    assert factory.create("my_patch").config_id == "my_patch"
    with pytest.raises(KeyError):
        factory.create("missing")


def test_a_run_through_a_cached_model_patches_the_same_way(model: HookedTransformer, tmp_path: Path) -> None:
    # Patching tokenizes, reads weights and runs with hooks through the wrapper; only its two cached
    # forward passes go through the cache, so the metric must come out exactly as it does unwrapped.
    def patch(target) -> torch.Tensor:
        patching_input = PatchingInput(_last_layer_resid_patch_config(), SOURCE_PROMPT, TARGET_PROMPT)
        collector = Patcher(target).apply(patching_input, return_collector=True, verbose=False)
        assert collector is not None
        return _metric(collector)

    direct = patch(model)
    cached_model = CachedModel(model, str(tmp_path))

    assert torch.equal(patch(cached_model), direct)
    assert len(list(tmp_path.rglob("*.pt"))) == 2  # one per prompt
    assert torch.equal(patch(cached_model), direct)  # now served from disk


# --------------------------------------------------------------------------------------------------
# The plan a run would execute
# --------------------------------------------------------------------------------------------------
def _every_layer_patch_config() -> PatchingConfig:
    """Replace the residual stream of every layer in turn — one patched run per layer."""
    element = PatchingElementConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelLocation.LAYER],
        type=PatchingType.REPLACE,
    )
    return _patch_config(element, config_id="every_layer_resid_patch")


def test_a_plan_counts_the_patch_runs_and_their_batches(model: HookedTransformer, monkeypatch, tmp_path: Path) -> None:
    save_dir = tmp_path / "planned"
    patching_input = PatchingInput(_every_layer_patch_config(), SOURCE_PROMPT, TARGET_PROMPT)

    forbid_model_runs(monkeypatch, model)
    plan = Patcher(model, batch_size=2).plan(patching_input, save_dir=str(save_dir), verbose=False)

    assert not save_dir.exists()
    assert plan.steps == model.cfg.n_layers
    assert plan.batches == (model.cfg.n_layers + 1) // 2
    # One patched run per batch, on top of the two cached runs the input prepares.
    assert plan.forward_passes == plan.batches + 2
    assert [(label, prompt) for label, prompt, _ in plan.prompts] == [
        ("source", SOURCE_PROMPT),
        ("target", TARGET_PROMPT),
    ]
    assert f"{plan.steps} patch runs in {plan.batches} batches" in str(plan)


def test_a_plan_describes_the_run_that_follows_it(model: HookedTransformer, tmp_path: Path) -> None:
    save_dir = tmp_path / "patch_run"
    elements = _mlp_mhsa_pattern_elements(_MULTI_PATCH_LAYER, _MULTI_PATCH_HEAD)
    patching_input = PatchingInput(_patch_config(*elements), SOURCE_PROMPT, TARGET_PROMPT)

    plan = Patcher(model).plan(patching_input, save_dir=str(save_dir), verbose=False)
    # The same input object: planning resolves its indices, the run still caches both prompts.
    collector = Patcher(model).apply(patching_input, save_dir=str(save_dir), return_collector=True, verbose=False)
    assert collector is not None

    assert sorted(path.name for path in save_dir.iterdir()) == sorted(plan.outputs)
    # One index per patch, each with its own file.
    assert len(plan.index) == len(elements)
    assert [name for name in plan.outputs if name.startswith("index-")] == [
        f"index-{i}.csv" for i in range(len(elements))
    ]
    for planned, idx in zip(plan.index, patching_input.index, strict=True):
        pd.testing.assert_frame_equal(planned, idx.to_df())
    assert plan.steps == len(collector.result)


# --------------------------------------------------------------------------------------------------
# Visualization — the run's own grid, drawn from result.csv's columns
# --------------------------------------------------------------------------------------------------
_FIRST_THREE_LAYERS = Filter(inclusion_terms=[{ModelLocation.LAYER: {"range": [0, 2]}}])


def _layer_position_config(
    metric: PatchingMetric = PatchingMetric.LOGIT_DIFF, index_filter: Filter = _FIRST_THREE_LAYERS
) -> PatchingConfig:
    element = PatchingElementConfig(
        activation=ModelLayerLocation.RESIDUAL_POST,
        location=[ModelLocation.LAYER, ModelLocation.POSITION],
        filter=index_filter,
        type=PatchingType.REPLACE,
    )
    return PatchingConfig(
        config_id="sweep", metric=metric, patch=[SinglePatchingConfig(source=element, target=element)]
    )


def _draw(
    model: HookedTransformer,
    config: PatchingConfig,
    save_dir: Path,
    source: str = SOURCE_PROMPT,
    target: str = TARGET_PROMPT,
) -> RenderedFigure:
    Patcher(model).apply(PatchingInput(config, source, target), save_dir=str(save_dir), verbose=False)
    return rendered_figure(save_dir)


def test_a_patching_run_writes_an_interactive_result_page(model: HookedTransformer, tmp_path: Path) -> None:
    save_dir = tmp_path / "patch_run"
    figure = _draw(model, _layer_position_config(), save_dir)

    assert "Plotly.newPlot" in (save_dir / "result.html").read_text(encoding="utf-8")
    assert figure.title.startswith("Patching — logit_diff (config: sweep)")


def test_a_sweep_is_drawn_with_positions_across_and_layers_down(model: HookedTransformer, tmp_path: Path) -> None:
    figure = _draw(model, _layer_position_config(), tmp_path / "patch_run")

    assert figure.kind == "heatmap"
    assert (figure.axis_title("xaxis"), figure.axis_title("yaxis")) == ("position", "layer")
    # plotly draws the first row at the bottom, which is where layer 0 belongs.
    assert figure.ticks("yaxis") == ["0", "1", "2"]
    assert len(figure.trace["z"]) == 3
    assert len(figure.trace["z"][0]) == model.to_tokens(TARGET_PROMPT).shape[1]


def test_the_position_axis_is_labelled_with_the_target_prompts_tokens(model: HookedTransformer, tmp_path: Path) -> None:
    # The patched run is the target run, so it is the target's tokens that name the positions — the
    # two prompts differ at position 4, which is what makes the choice observable.
    ticks = _draw(model, _layer_position_config(), tmp_path / "patch_run").ticks("xaxis")

    assert ticks[4] == '4: " Spain"'
    assert '4: " France"' not in ticks


def test_the_colour_is_centred_on_the_value_an_ineffective_patch_would_produce(
    model: HookedTransformer, tmp_path: Path
) -> None:
    save_dir = tmp_path / "patch_run"
    patching_input = PatchingInput(_layer_position_config(), SOURCE_PROMPT, TARGET_PROMPT)
    Patcher(model).apply(patching_input, save_dir=str(save_dir), verbose=False)

    figure = rendered_figure(save_dir)
    trace = figure.trace
    baseline = float(patching_input.target_baseline)

    assert trace["zmin"] == pytest.approx(2 * baseline - trace["zmax"])
    assert trace["colorbar"]["title"]["text"] == "logit_diff"
    # Nothing is printed inside a patching cell; the metric is the colour and the rest is hover.
    assert "text" not in trace


def test_patching_a_prompt_with_itself_leaves_the_scale_intact(model: HookedTransformer, tmp_path: Path) -> None:
    # Every patch writes back what was already there, so every cell scores the same and the colour
    # range has no spread of its own to take.
    figure = _draw(model, _layer_position_config(), tmp_path / "no_op", source=TARGET_PROMPT)

    assert {value for row in figure.trace["z"] for value in row} == {0.0}
    assert (figure.trace["zmin"], figure.trace["zmax"]) == (-1.0, 1.0)


def test_an_axis_the_run_does_not_vary_over_is_dropped(model: HookedTransformer, tmp_path: Path) -> None:
    # One layer over every position: the layer offers no comparison, so it leaves the axes entirely.
    config = _layer_position_config(index_filter=Filter(inclusion_terms=[{ModelLocation.LAYER: 0}]))

    figure = _draw(model, config, tmp_path / "one_layer")

    assert figure.kind == "bar"
    assert figure.axis_title("xaxis") == "position"
    assert figure.axis_title("yaxis") == "logit_diff"
    assert len(figure.trace["y"]) == model.to_tokens(TARGET_PROMPT).shape[1]


def test_a_sparse_index_leaves_holes_in_the_grid(model: HookedTransformer, tmp_path: Path) -> None:
    # Two correlated points rather than a full grid: the pairs the run never visited stay empty.
    config = _layer_position_config(
        index_filter=Filter(
            inclusion_terms=[
                {ModelLocation.LAYER: 0, ModelLocation.POSITION: 1},
                {ModelLocation.LAYER: 1, ModelLocation.POSITION: 2},
            ]
        )
    )

    figure = _draw(model, config, tmp_path / "sparse")

    assert figure.kind == "heatmap"
    assert [[cell is None for cell in row] for row in figure.trace["z"]] == [[False, True], [True, False]]


def test_explicit_coordinates_are_drawn_as_a_bar_labelled_by_coordinate(
    model: HookedTransformer, tmp_path: Path
) -> None:
    # An explicit index is a list of points, not a grid, so there is no second axis to lay out.
    coordinates = [
        {ModelLocation.LAYER: -2, ModelLocation.POSITION: -1},
        {ModelLocation.LAYER: -1, ModelLocation.POSITION: -1},
    ]

    figure = _draw(model, _explicit_patch_config(coordinates, coordinates), tmp_path / "coordinates")

    assert figure.kind == "bar"
    assert len(figure.trace["y"]) == 2
    assert figure.ticks("xaxis")[0].startswith("resid.post · layer")


def test_the_axis_falls_back_to_the_target_when_the_source_does_not_span_it(
    model: HookedTransformer, tmp_path: Path
) -> None:
    # One source coordinate read into two target coordinates: the source is repeated, so its columns
    # are half empty in the result and cannot name the axis the target laid out.
    source = [{ModelLocation.LAYER: -1, ModelLocation.POSITION: -1}]
    target = [
        {ModelLocation.LAYER: -1, ModelLocation.POSITION: -2},
        {ModelLocation.LAYER: -1, ModelLocation.POSITION: -1},
    ]

    save_dir = tmp_path / "uneven"
    figure = _draw(model, _explicit_patch_config(source, target), save_dir)

    assert pd.read_csv(save_dir / "result.csv")["source_position"].isna().any()
    positions = model.to_tokens(TARGET_PROMPT).shape[1]
    assert figure.ticks("xaxis")[0].endswith(f"position {positions - 2}")


def test_a_single_coordinate_run_still_writes_a_page(model: HookedTransformer, tmp_path: Path) -> None:
    # One row has no axis to vary over at all, so the grid degenerates to a single bar.
    coordinates = [{ModelLocation.LAYER: -1, ModelLocation.POSITION: -1}]

    figure = _draw(model, _explicit_patch_config(coordinates, coordinates), tmp_path / "one_coordinate")

    assert figure.kind == "bar"
    assert len(figure.trace["y"]) == 1
    assert figure.title.startswith("Patching — normalized_logit_diff")


def test_a_grouped_location_does_not_break_the_page(model: HookedTransformer, tmp_path: Path) -> None:
    # Grouping puts a list of heads in one index cell; it has to survive being turned into a label.
    element = PatchingElementConfig(
        activation=ModelLayerLocation.ATTENTION_Z,
        location=[ModelLocation.LAYER, ModelLocation.HEAD],
        filter=Filter(inclusion_terms=[{ModelLocation.LAYER: -1, ModelLocation.HEAD: [0, 1, 2]}]),
        group=Group({ModelLocation.HEAD: [0, 1]}),
        type=PatchingType.REPLACE,
    )
    config = PatchingConfig(
        metric=PatchingMetric.LOGIT_DIFF, patch=[SinglePatchingConfig(source=element, target=element)]
    )

    figure = _draw(model, config, tmp_path / "grouped")

    assert figure.kind == "bar"
    assert any("[0, 1]" in tick for tick in figure.ticks("xaxis"))


def test_the_other_patched_elements_are_named_on_the_page(model: HookedTransformer, tmp_path: Path) -> None:
    # A companion element patched identically on every run is constant, so it belongs in the subtitle
    # rather than in every cell's hover.
    companion = PatchingElementConfig(
        activation=ModelLayerLocation.ATTENTION_PATTERN,
        location=[ModelLocation.LAYER, ModelLocation.HEAD],
        filter=Filter(inclusion_terms=[{ModelLocation.LAYER: 0, ModelLocation.HEAD: 0}]),
        type=PatchingType.REPLACE,
    )
    config = _layer_position_config()
    config.patch.append(SinglePatchingConfig(source=companion, target=companion))

    figure = _draw(model, config, tmp_path / "multi_element")

    assert "also patched: attn.pattern · layer 0 · head 0" in figure.title


def test_the_base_collector_skips_the_html(model: HookedTransformer, tmp_path: Path) -> None:
    # The base collector persists the CSVs only; visualization is opt-in via the subclass.
    save_dir = tmp_path / "no_viz"
    collector = Patcher(model, collector_class=PatchingCollector).apply(
        PatchingInput(_layer_position_config(), SOURCE_PROMPT, TARGET_PROMPT),
        save_dir=str(save_dir),
        return_collector=True,
        verbose=False,
    )

    assert collector is not None
    assert type(collector) is PatchingCollector
    assert (save_dir / "result.csv").exists()
    assert not (save_dir / "result.html").exists()


def test_a_plan_for_the_base_collector_promises_no_html(model: HookedTransformer, tmp_path: Path) -> None:
    save_dir = tmp_path / "no_viz_plan"
    patching_input = PatchingInput(_layer_position_config(), SOURCE_PROMPT, TARGET_PROMPT)
    patcher = Patcher(model, collector_class=PatchingCollector)

    plan = patcher.plan(patching_input, save_dir=str(save_dir), verbose=False)
    patcher.apply(patching_input, save_dir=str(save_dir), verbose=False)

    assert "result.html" not in plan.outputs
    assert sorted(path.name for path in save_dir.iterdir()) == sorted(plan.outputs)
