"""The ``transformer_lens`` hook each activation is cached under."""

from entoscope.model.config import ModelLayerLocation

TRANSFORMER_ACT_NAME: dict[ModelLayerLocation, str] = {
    ModelLayerLocation.EMBEDDING: "hook_embed",
    ModelLayerLocation.RESIDUAL_PRE: "hook_resid_pre",
    ModelLayerLocation.RESIDUAL_MID: "hook_resid_mid",
    ModelLayerLocation.RESIDUAL_POST: "hook_resid_post",
    ModelLayerLocation.ATTENTION_IN: "attn.hook_in",
    ModelLayerLocation.ATTENTION_Q: "attn.hook_q",
    ModelLayerLocation.ATTENTION_K: "attn.hook_k",
    ModelLayerLocation.ATTENTION_V: "attn.hook_v",
    ModelLayerLocation.ATTENTION_PATTERN: "attn.hook_pattern",
    ModelLayerLocation.ATTENTION_Z: "attn.hook_z",
    ModelLayerLocation.ATTENTION_OUT_PRE_NORM: "attn.hook_out",
    ModelLayerLocation.ATTENTION_OUT: "hook_attn_out",
    ModelLayerLocation.MLP_IN: "mlp.hook_in",
    ModelLayerLocation.MLP_PRE: "mlp.hook_pre",
    ModelLayerLocation.MLP_PRE_LINEAR: "mlp.hook_pre_linear",
    ModelLayerLocation.MLP_POST: "mlp.hook_post",
    ModelLayerLocation.MLP_OUT_PRE_NORM: "mlp.hook_out",
    ModelLayerLocation.MLP_OUT: "hook_mlp_out",
}


def hook_name(activation: ModelLayerLocation, layer: int | None) -> str:
    """Name the hook an activation is cached under.

    Args:
        activation: The activation.
        layer: The block it is read in, ``None`` for ``embed``.

    Returns:
        The ``transformer_lens`` hook name, such as ``blocks.3.hook_resid_post``.
    """
    name = TRANSFORMER_ACT_NAME[activation]
    return name if layer is None else f"blocks.{layer}.{name}"
