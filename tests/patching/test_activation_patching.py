import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from transformer_lens import HookedTransformer

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
        {"metric": "logit_diff", "patch": [{"source": element, "target": element}]},
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
        {"metric": "logit_diff", "patch": [{"source": source, "target": target}]},
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

    assert torch.allclose(result.squeeze(), torch.tensor(2.19863224029541))


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

    assert torch.allclose(result.squeeze(), torch.tensor(1.2414560317993164))


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

    assert torch.allclose(scaled, torch.tensor(2.0231308937072754))
    assert torch.allclose(unscaled, torch.tensor(2.19863224029541))
    assert not torch.allclose(scaled, unscaled)


def test_artifacts_saved_to_disk(model: HookedTransformer, tmp_path: Path) -> None:
    torch.manual_seed(42)

    source_tokens = model.to_tokens(SOURCE_PROMPT)
    target_tokens = model.to_tokens(TARGET_PROMPT)

    with torch.no_grad():
        source_logits, source_cache = model.run_with_cache(source_tokens)
        target_logits, target_cache = model.run_with_cache(target_tokens)

    config = _last_layer_resid_patch_config()
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
