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
