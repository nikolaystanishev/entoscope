"""Read the residual-stream / MLP tensors and compute the decomposed attention from a cache.

These are the model-side inputs the (vendored) Information Flow Routes contribution math needs.
The residual and MLP readers are thin wrappers over ``ActivationCache``; ``decomposed_attn`` is a
reimplementation of the LM Transparency Tool's ``TransformerLensTransparentLlm.decomposed_attn``
einsum — reimplemented (not vendored) because that module hard-imports ``streamlit`` and loads its
own model, whereas intex passes an already-loaded ``HookedTransformer`` around.
"""

import einops
import torch
from jaxtyping import Float
from transformer_lens import ActivationCache, HookedTransformer


def residual_in(cache: ActivationCache, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
    return cache["resid_pre", layer]


def residual_after_attn(cache: ActivationCache, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
    return cache["resid_mid", layer]


def residual_out(cache: ActivationCache, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
    return cache["resid_post", layer]


def ffn_out(cache: ActivationCache, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
    return cache["mlp_out", layer]


@torch.no_grad()
def decomposed_attn(
    model: HookedTransformer,
    cache: ActivationCache,
    layer: int,
    batch_i: int = 0,
) -> Float[torch.Tensor, "pos key_pos head d_model"]:
    """Per (query_pos, key_pos, head) attention contribution to the residual stream, in d_model space.

    Reconstructs each head's output written to the residual stream, split by the key position it came
    from, before the residual add (no layernorm is applied — the attention output is added to the
    residual stream unnormalized). Summing over ``key_pos`` and ``head`` recovers ``attn_out`` minus
    the ``b_O`` bias. Handles grouped-query attention by repeating the value heads across their group.
    """
    hook_v = cache["v", layer][batch_i]  # [key_pos, head_kv, d_head]
    b_v = model.blocks[layer].attn.b_V  # [head, d_head]

    num_head_groups = b_v.shape[-2] // hook_v.shape[-2]
    hook_v = hook_v.repeat_interleave(num_head_groups, dim=-2)  # [key_pos, head, d_head]

    v = hook_v + b_v
    pattern = cache["pattern", layer][batch_i].to(v.dtype)  # [head, query_pos, key_pos]

    z = einops.einsum(
        v,
        pattern,
        "key_pos head d_head, head query_pos key_pos -> query_pos key_pos head d_head",
    )
    return einops.einsum(
        z,
        model.W_O[layer],  # [head, d_head, d_model]
        "query_pos key_pos head d_head, head d_head d_model -> query_pos key_pos head d_model",
    )
