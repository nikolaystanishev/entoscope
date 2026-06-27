import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
from transformer_lens import HookedTransformer

from intex.model.config import ModelLayerLocation
from intex.operation.config import ConfigFactory
from intex.patching.config import PatchingConfig
from intex.patching.input import PatchingInput
from intex.patching.patcher import Patcher

SOURCE_PROMPT = "The capital of France is"
TARGET_PROMPT = "The capital of Spain is"


@pytest.fixture(scope="module")
def model() -> HookedTransformer:
    model = HookedTransformer.from_pretrained("gpt2")
    model.eval()
    return model


def _last_layer_resid_patch_config() -> PatchingConfig:
    """Replace the last layer's full residual stream (all positions) source -> target."""
    element = {
        "activation_name": "resid.post",
        "location": ["layer"],
        "filter": [{"layer": -1}],
        "type": "replace",
    }
    return PatchingConfig(
        "last_layer_resid_patch",
        {"metric": "normalized_logit_diff", "patch": [{"source": element, "target": element}]},
    )


def _last_layer_resid_zero_ablation_config() -> PatchingConfig:
    """Zero-ablate the last layer's full residual stream (all positions): set every value to 0."""
    source = {"type": "value_replace"}  # produces zeros
    target = {
        "activation_name": "resid.post",
        "location": ["layer"],
        "filter": [{"layer": -1}],
        "type": "replace",
    }  # writes them into the activation
    return PatchingConfig(
        "last_layer_resid_zero_ablation",
        {"metric": "logit_diff", "patch": [{"source": source, "target": target}]},
    )


def _resid_value_scale_config(scale: float, layer: int) -> PatchingConfig:
    """Scale a layer's source residual stream (all positions) by ``scale``, then write it into the target.

    The scaled layer is intentionally not the final one: ``resid.post`` of the last layer feeds straight
    into ``ln_final``, whose normalization cancels any uniform scaling, so the scale would have no effect
    on the logits. At an intermediate layer the scaled residual is carried forward through the skip
    connections, so the scale genuinely changes the output.
    """
    source = {
        "activation_name": "resid.post",
        "location": ["layer"],
        "filter": [{"layer": layer}],
        "type": "value_scale",
        "value": scale,
    }
    target = {
        "activation_name": "resid.post",
        "location": ["layer"],
        "filter": [{"layer": layer}],
        "type": "replace",
    }
    return PatchingConfig(
        "resid_value_scale",
        {"metric": "kl_diff", "patch": [{"source": source, "target": target}]},
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
    mlp = {
        "activation_name": "mlp.post",
        "location": ["layer"],
        "filter": [{"layer": layer}],
        "type": "replace",
    }
    mhsa = {
        "activation_name": "attn.q",
        "location": ["layer"],
        "filter": [{"layer": layer}],
        "type": "replace",
    }
    pattern = {
        "activation_name": "attn.pattern",
        "location": ["layer", "head"],
        "filter": [{"layer": layer, "head": head}],
        "type": "replace",
    }
    return PatchingConfig(
        "mlp_mhsa_pattern_patch",
        {
            "metric": "logit_diff",
            "patch": [
                {"source": mlp, "target": mlp},
                {"source": mhsa, "target": mhsa},
                {"source": pattern, "target": pattern},
            ],
        },
    )


def test_activation_patching_from_source_into_target(model: HookedTransformer) -> None:
    torch.manual_seed(42)

    source_tokens = model.to_tokens(SOURCE_PROMPT)
    target_tokens = model.to_tokens(TARGET_PROMPT)
    # A full residual replacement requires matching sequence lengths between the two prompts.
    assert source_tokens.shape == target_tokens.shape

    with torch.no_grad():
        source_logits, source_cache = model.run_with_cache(source_tokens)
        target_logits, target_cache = model.run_with_cache(target_tokens)

    # Two distinct answer tokens for the logit-diff metric ("the capital" answers).
    source_generation = model.to_single_token(" Paris")
    target_generation = model.to_single_token(" Madrid")

    patching_input = PatchingInput(
        source_tokens=source_tokens,
        target_tokens=target_tokens,
        source_logits=source_logits,
        target_logits=target_logits,
        source_generation=source_generation,
        target_generation=target_generation,
        source_cache=source_cache,
        target_cache=target_cache,
        config=_last_layer_resid_patch_config(),
    )

    collector = Patcher(model).apply(patching_input, return_collector=True, verbose=False)

    # One patch (the single last layer) -> a one-element result tensor.
    assert collector is not None
    result = collector.result
    assert result.shape == (1,)

    assert torch.allclose(result.squeeze(), torch.tensor(1.0))


def test_zero_ablation_patching(model: HookedTransformer) -> None:
    torch.manual_seed(42)

    # Zero ablation operates on a single run: there is no real source to patch from, so the same
    # prompt drives both source and target tokens and no source activations are needed.
    tokens = model.to_tokens(SOURCE_PROMPT)

    with torch.no_grad():
        target_logits, target_cache = model.run_with_cache(tokens)

    source_generation = model.to_single_token(" Paris")
    target_generation = model.to_single_token(" Madrid")

    patching_input = PatchingInput(
        source_tokens=None,
        target_tokens=tokens,
        source_logits=None,
        target_logits=target_logits,
        source_generation=source_generation,
        target_generation=target_generation,
        source_cache=None,
        target_cache=target_cache,
        config=_last_layer_resid_zero_ablation_config(),
    )

    collector = Patcher(model).apply(patching_input, return_collector=True, verbose=False)

    # One patch (the single last layer) -> a one-element result tensor.
    assert collector is not None
    result = collector.result
    assert result.shape == (1,)

    # atol absorbs cross-platform float32 drift in the matmuls (arm64/x86) while staying tight.
    assert torch.allclose(result.squeeze(), torch.tensor(1.2414560317993164), atol=1e-2)


# Intermediate layer whose residual stream is scaled; see ``_resid_value_scale_config`` for why it is
# not the final layer.
_VALUE_SCALE_LAYER = 6


def _run_value_scale(model: HookedTransformer, scale: float) -> torch.Tensor:
    source_tokens = model.to_tokens(SOURCE_PROMPT)
    target_tokens = model.to_tokens(TARGET_PROMPT)
    # Scaling the source residual and writing it into the target requires matching sequence lengths.
    assert source_tokens.shape == target_tokens.shape

    with torch.no_grad():
        source_logits, source_cache = model.run_with_cache(source_tokens)
        target_logits, target_cache = model.run_with_cache(target_tokens)

    source_generation = model.to_single_token(" Paris")
    target_generation = model.to_single_token(" Madrid")

    patching_input = PatchingInput(
        source_tokens=source_tokens,
        target_tokens=target_tokens,
        source_logits=source_logits,
        target_logits=target_logits,
        source_generation=source_generation,
        target_generation=target_generation,
        source_cache=source_cache,
        target_cache=target_cache,
        config=_resid_value_scale_config(scale, _VALUE_SCALE_LAYER),
    )

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

    source_tokens = model.to_tokens(SOURCE_PROMPT)
    target_tokens = model.to_tokens(TARGET_PROMPT)
    # Replacing full activations (and a head's pattern) from source requires matching sequence lengths.
    assert source_tokens.shape == target_tokens.shape

    with torch.no_grad():
        source_logits, source_cache = model.run_with_cache(source_tokens)
        target_logits, target_cache = model.run_with_cache(target_tokens)

    source_generation = model.to_single_token(" Paris")
    target_generation = model.to_single_token(" Madrid")

    patching_input = PatchingInput(
        source_tokens=source_tokens,
        target_tokens=target_tokens,
        source_logits=source_logits,
        target_logits=target_logits,
        source_generation=source_generation,
        target_generation=target_generation,
        source_cache=source_cache,
        target_cache=target_cache,
        config=_mlp_mhsa_pattern_patch_config(_MULTI_PATCH_LAYER, _MULTI_PATCH_HEAD),
    )

    collector = Patcher(model).apply(patching_input, return_collector=True, verbose=False)

    # The config carries three patches; each becomes its own index, targeting -- in order -- the MLP
    # post-activation, the MHSA queries, and the attention pattern.
    indices = patching_input.get_index()
    assert [idx.config.target.activation_name for idx in indices] == [
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
    # logit diff of any single component patched on its own (mlp.post: -3.337, attn.q: -3.914,
    # attn.pattern: -3.913), confirming all three patches take effect simultaneously. The atol absorbs
    # cross-platform float32 drift in the matmuls while staying well inside that ~0.02 gap.
    assert torch.allclose(result.squeeze(), torch.tensor(-3.316533088684082), atol=1e-2)


def test_artifacts_saved_to_disk(model: HookedTransformer, tmp_path: Path) -> None:
    torch.manual_seed(42)

    source_tokens = model.to_tokens(SOURCE_PROMPT)
    target_tokens = model.to_tokens(TARGET_PROMPT)

    with torch.no_grad():
        source_logits, source_cache = model.run_with_cache(source_tokens)
        target_logits, target_cache = model.run_with_cache(target_tokens)

    # Build the config through ConfigFactory by persisting it to a YAML file and loading it back,
    # mirroring how configs are sourced in production rather than constructing PatchingConfig directly.
    element = {
        "activation_name": "resid.post",
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
    patching_input = PatchingInput(
        source_tokens=source_tokens,
        target_tokens=target_tokens,
        source_logits=source_logits,
        target_logits=target_logits,
        source_generation=model.to_single_token(" Paris"),
        target_generation=model.to_single_token(" Madrid"),
        source_cache=source_cache,
        target_cache=target_cache,
        config=config,
    )

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
    assert len(saved_config["patching_configs"]) == 1
    assert saved_config["patching_configs"][0]["source"]["type"] == "replace"
    assert saved_config["patching_configs"][0]["target"]["type"] == "replace"

    # index-0.csv carries one row per result entry, with the source/target/index columns.
    index_df = pd.read_csv(index_path)
    assert len(index_df) == collector.result.numel()
    assert any(col.startswith("source_") for col in index_df.columns)
    assert any(col.startswith("target_") for col in index_df.columns)
    assert any(col.startswith("index_") for col in index_df.columns)
