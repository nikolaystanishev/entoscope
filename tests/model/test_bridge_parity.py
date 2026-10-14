"""The two backends, checked against each other rather than each on its own.

The rest of the suite runs every test against a bridge and a ``HookedTransformer`` separately, which
says each one works but not that they agree. A bridge in compatibility mode is processed the way
``HookedTransformer.from_pretrained`` processes its weights, so on the same model the two are the
same numbers — and that is what makes a result portable between them.
"""

import pytest
import torch

from entoscope.lens.config import LensConfig, LensMethodConfig
from entoscope.lens.input import LensInput
from entoscope.lens.lens import Lens
from entoscope.lens.method.logit_lens import _attention_scale
from entoscope.model.config import ModelLayerLocation, ModelLocation, ModelLocatorConfig
from entoscope.model.weight import attention_out_weight, mlp_weight

from ..conftest import HookedTransformer, TransformerBridge, _gpt2_hooked

pytestmark = pytest.mark.skipif(
    TransformerBridge is None or HookedTransformer is None, reason="transformer_lens has only one of the backends"
)

PROMPT = "The capital of France is"

# What "the same numbers" can mean for a forward pass. The two backends run the same arithmetic on
# the same weights, but not through the same kernels — the matmuls are issued differently, and a
# folded ``ln_final`` is HuggingFace's LayerNorm on one side and ``LayerNormPre`` on the other — so
# the last places of a float32 value are the build's to decide, and CI has seen two runners disagree
# with three on a single logit. A parity break worth catching is orders of magnitude wider.
TO_FLOAT32 = {"rtol": 1e-4, "atol": 1e-3}


@pytest.fixture(scope="module")
def compatible_bridge():
    model = TransformerBridge.boot_transformers("gpt2", device="cpu")
    model.enable_compatibility_mode(disable_warnings=True)
    model.eval()
    return model


@pytest.fixture(scope="module")
def hooked():
    return _gpt2_hooked()


def test_the_weights_a_lens_reads_are_the_same(compatible_bridge, hooked):
    assert torch.equal(compatible_bridge.W_U, hooked.W_U)
    assert torch.equal(compatible_bridge.b_U, hooked.b_U)
    assert torch.equal(compatible_bridge.W_E, hooked.W_E)

    for layer in range(hooked.cfg.n_layers):
        assert torch.equal(mlp_weight(compatible_bridge, layer, "W_in"), mlp_weight(hooked, layer, "W_in"))
        assert torch.equal(mlp_weight(compatible_bridge, layer, "W_out"), mlp_weight(hooked, layer, "W_out"))
        assert torch.equal(attention_out_weight(compatible_bridge, layer), attention_out_weight(hooked, layer))
        for name in ("W_Q", "W_K", "W_V"):
            bridge_attention = getattr(compatible_bridge.blocks[layer].attn, name)
            assert torch.equal(bridge_attention, getattr(hooked.blocks[layer].attn, name))


def test_the_attention_scale_resolves_the_same(compatible_bridge, hooked):
    # A bridge leaves cfg.attn_scale at its -1.0 sentinel, which would silently negate a QK reading.
    assert compatible_bridge.cfg.attn_scale == -1.0
    assert _attention_scale(compatible_bridge) == _attention_scale(hooked) == hooked.cfg.attn_scale


def test_a_forward_pass_produces_the_same_logits_and_activations(compatible_bridge, hooked):
    tokens = hooked.to_tokens(PROMPT)
    assert torch.equal(compatible_bridge.to_tokens(PROMPT), tokens)

    names = [f"blocks.{layer}.hook_resid_post" for layer in range(hooked.cfg.n_layers)]
    with torch.no_grad():
        bridge_logits, bridge_cache = compatible_bridge.run_with_cache(tokens, names_filter=names)
        hooked_logits, hooked_cache = hooked.run_with_cache(tokens, names_filter=names)

    assert torch.allclose(bridge_logits, hooked_logits, **TO_FLOAT32)
    for name in names:
        assert torch.allclose(bridge_cache[name], hooked_cache[name], **TO_FLOAT32)


@pytest.mark.parametrize(
    "activation",
    [ModelLayerLocation.RESIDUAL_POST, ModelLayerLocation.MLP_POST, ModelLayerLocation.ATTENTION_Z],
)
def test_a_lens_run_reads_the_same_tokens_from_either_backend(compatible_bridge, hooked, activation):
    config = LensConfig(
        method=LensMethodConfig(type="logit_lens"),
        top_k=3,
        source=ModelLocatorConfig(activation=activation, location=[ModelLocation.LAYER, ModelLocation.POSITION]),
    )

    def run(model):
        collector = Lens(model).apply(LensInput(config, PROMPT), return_collector=True, verbose=False)
        return collector.result

    bridge_result, hooked_result = run(compatible_bridge), run(hooked)

    # The ranking is identical; the scores only agree to float32. The two read the same residuals
    # through the same weights, but a folded ``ln_final`` is HuggingFace's LayerNorm on one side and
    # ``LayerNormPre`` on the other, and the difference survives the softmax over the vocabulary.
    assert bridge_result["token_0"].to_list() == hooked_result["token_0"].to_list()
    assert bridge_result["probability_0"].to_list() == pytest.approx(hooked_result["probability_0"].to_list(), rel=1e-4)
