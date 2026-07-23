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
from intex.patching.config import (
    PatchingConfig,
    PatchingElementConfig,
    PatchingMetric,
    PatchingType,
    SinglePatchingConfig,
)
from intex.patching.input import PatchingInput
from intex.patching.patcher import Patcher

SOURCE_PROMPT = "The capital of France is"
TARGET_PROMPT = "The capital of Spain is"


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
        metric=PatchingMetric.KL_DIFF,
        patch=[SinglePatchingConfig(source=source, target=target)],
    )


def _mlp_mhsa_pattern_patch_config(layer: int, head: int) -> PatchingConfig:
    """Patch three components of one layer at once, each replaced from the source run:

    - the MLP post-activation neurons (``mlp.post``),
    - the multi-head self-attention queries (``attn.q``), and
    - a single attention head's pattern (``attn.pattern`` filtered to ``head``).

    The three patches live in the same config's ``patch`` list, so the patcher applies all of their hooks
    together on a single forward pass; the resulting (single-element) metric reflects their combined
    effect rather than any one in isolation.
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
    return PatchingConfig(
        config_id="mlp_mhsa_pattern_patch",
        metric=PatchingMetric.LOGIT_DIFF,
        patch=[
            SinglePatchingConfig(source=mlp, target=mlp),
            SinglePatchingConfig(source=mhsa, target=mhsa),
            SinglePatchingConfig(source=pattern, target=pattern),
        ],
    )


def test_activation_patching_from_source_into_target(model: HookedTransformer) -> None:
    torch.manual_seed(42)

    # A full residual replacement requires matching sequence lengths between the two prompts.
    assert model.to_tokens(SOURCE_PROMPT).shape == model.to_tokens(TARGET_PROMPT).shape

    patching_input = PatchingInput(_last_layer_resid_patch_config(), SOURCE_PROMPT, TARGET_PROMPT)

    collector = Patcher(model).apply(patching_input, return_collector=True, verbose=False)

    # One patch (the single last layer) -> a one-element result tensor.
    assert collector is not None
    result = collector.result
    assert result.shape == (1,)

    assert torch.allclose(result.squeeze(), torch.tensor(1.0))


def test_caches_restricted_to_used_activations(model: HookedTransformer) -> None:
    patching_input = PatchingInput(_last_layer_resid_patch_config(), SOURCE_PROMPT, TARGET_PROMPT).prepare(model)

    # Only the last layer's residual is read; both caches are filtered down to it.
    expected = set().union(*(idx.source_activation_names() for idx in patching_input.index))
    assert expected == {f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"}
    assert set(patching_input.source_cache.keys()) == expected
    assert set(patching_input.target_cache.keys()) == expected

    _, full_cache = model.run_with_cache(patching_input.source_tokens)
    assert set(patching_input.source_cache.keys()) < set(full_cache.keys())


def test_zero_ablation_patching(model: HookedTransformer) -> None:
    torch.manual_seed(42)

    # Zero ablation writes zeros into the target run, so the source run's activations are never read;
    # its forward pass only supplies the source generation token that the logit-diff metric compares
    # against the target one. Distinct prompts keep those two tokens distinct.
    patching_input = PatchingInput(_last_layer_resid_zero_ablation_config(), SOURCE_PROMPT, TARGET_PROMPT)

    collector = Patcher(model).apply(patching_input, return_collector=True, verbose=False)

    # One patch (the single last layer) -> a one-element result tensor.
    assert collector is not None
    result = collector.result
    assert result.shape == (1,)

    # atol absorbs cross-platform float32 drift in the matmuls (arm64/x86) while staying tight.
    assert torch.allclose(result.squeeze(), torch.tensor(-2.7828707695007324), atol=1e-2)


# Intermediate layer whose residual stream is scaled; see ``_resid_value_scale_config`` for why it is
# not the final layer.
_VALUE_SCALE_LAYER = 6


def _run_value_scale(model: HookedTransformer, scale: float) -> torch.Tensor:
    # Scaling the source residual and writing it into the target requires matching sequence lengths.
    assert model.to_tokens(SOURCE_PROMPT).shape == model.to_tokens(TARGET_PROMPT).shape

    patching_input = PatchingInput(_resid_value_scale_config(scale, _VALUE_SCALE_LAYER), SOURCE_PROMPT, TARGET_PROMPT)

    collector = Patcher(model).apply(patching_input, return_collector=True, verbose=False)

    # One patch (a single layer) -> a one-element result tensor.
    assert collector is not None
    result = collector.result
    assert result.shape == (1,)
    return result.squeeze()


def test_value_scale_patching(model: HookedTransformer) -> None:
    torch.manual_seed(42)

    # Scaling the source residual by 2x before writing it into the target changes the logit diff,
    # and is distinct from the unscaled (1x) patch below -- proving the scale actually takes effect.
    scaled = _run_value_scale(model, scale=2.0)
    unscaled = _run_value_scale(model, scale=1.0)

    # atol absorbs cross-platform float32 drift in the matmuls (~1e-3 between arm64/x86); it stays far
    # tighter than the ~1.07 gap that distinguishes the scaled run from the unscaled one below.
    assert torch.allclose(scaled, torch.tensor(-24.359235763549805), atol=1e-2)
    assert torch.allclose(unscaled, torch.tensor(-25.428979873657227), atol=1e-2)
    assert not torch.allclose(scaled, unscaled)


# Layer and head whose MLP, attention output, and head pattern are patched together below.
_MULTI_PATCH_LAYER = 5
_MULTI_PATCH_HEAD = 7


def test_multi_config_patching_mlp_mhsa_and_pattern(model: HookedTransformer) -> None:
    torch.manual_seed(42)

    # Replacing full activations (and a head's pattern) from source requires matching sequence lengths.
    assert model.to_tokens(SOURCE_PROMPT).shape == model.to_tokens(TARGET_PROMPT).shape

    patching_input = PatchingInput(
        _mlp_mhsa_pattern_patch_config(_MULTI_PATCH_LAYER, _MULTI_PATCH_HEAD), SOURCE_PROMPT, TARGET_PROMPT
    )

    collector = Patcher(model).apply(patching_input, return_collector=True, verbose=False)

    # The config carries three patches; each becomes its own index, targeting -- in order -- the MLP
    # post-activation, the MHSA queries, and the attention pattern.
    indices = patching_input.index
    assert [idx.target_config.activation for idx in indices] == [
        ModelLayerLocation.MLP_POST,
        ModelLayerLocation.ATTENTION_Q,
        ModelLayerLocation.ATTENTION_PATTERN,
    ]

    # All three patches are applied together on a single forward pass and the collector keys the result
    # off the first patch's (single) output coordinate -> a one-element tensor.
    assert collector is not None
    result = collector.result
    assert result.shape == (1,)

    # The exact logit diff produced by the three patches acting in combination. It differs from the
    # logit diff of any single component patched on its own (mlp.post: -0.0165, attn.q: -0.1038,
    # attn.pattern: -0.1061), confirming all three patches take effect simultaneously. The atol absorbs
    # cross-platform float32 drift in the matmuls (~1e-3 arm64/x86) while staying well inside the
    # ~0.008 gap to the nearest single-component value.
    assert torch.allclose(result.squeeze(), torch.tensor(-0.008522987365722656), atol=2e-3)


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
    return collector.result


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
    result_path = save_dir / "result.pt"
    index_path = save_dir / "index-0.csv"
    config_path = save_dir / "config.json"
    assert result_path.exists()
    assert index_path.exists()
    assert config_path.exists()

    # result.pt round-trips to exactly the in-memory result tensor.
    saved_result = torch.load(result_path)
    assert saved_result.shape == collector.result.shape == (1,)
    assert torch.equal(saved_result, collector.result)

    # config.json mirrors the patching config that produced the run.
    saved_config = json.loads(config_path.read_text())
    assert saved_config["config_id"] == "last_layer_resid_patch"
    assert saved_config["metric"] == "logit_diff"
    assert len(saved_config["patch"]) == 1
    assert saved_config["patch"][0]["source"]["type"] == "replace"
    assert saved_config["patch"][0]["target"]["type"] == "replace"

    # index-0.csv carries one row per result entry, with the source/target/index columns.
    index_df = pd.read_csv(index_path)
    assert len(index_df) == collector.result.numel()
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


def _run(model: HookedTransformer, config: PatchingConfig) -> torch.Tensor:
    collector = Patcher(model).apply(
        PatchingInput(config, SOURCE_PROMPT, TARGET_PROMPT), return_collector=True, verbose=False
    )
    assert collector is not None
    return collector.result


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
    assert torch.equal(torch.load(save_dir / "result.pt"), collector.result)
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
        return collector.result

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
    assert collector.result.shape == (1,)


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

    collector = Patcher(model).apply(
        PatchingInput(config, SOURCE_PROMPT, TARGET_PROMPT), return_collector=True, verbose=False
    )
    assert collector is not None
    assert torch.allclose(collector.result.squeeze(), torch.tensor(-2.7828707695007324), atol=1e-2)


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
    assert collector.result.shape == (2,)  # layers 0 and 2 survive the exclusion


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
        return collector.result

    direct = patch(model)
    cached_model = CachedModel(model, str(tmp_path))

    assert torch.equal(patch(cached_model), direct)
    assert len(list(tmp_path.rglob("*.pt"))) == 2  # one per prompt
    assert torch.equal(patch(cached_model), direct)  # now served from disk
