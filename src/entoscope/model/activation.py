"""The transformer activation name each activation is cached under, and how a hook writing one hands it on."""

from collections.abc import Iterable

import torch

from entoscope.model.base import Model
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


def transformer_activation(activation: ModelLayerLocation, layer: int | None) -> str:
    """Name an activation as ``transformer_lens`` caches it.

    Args:
        activation: The activation.
        layer: The block it is read in, ``None`` for ``embed``.

    Returns:
        The transformer activation name, such as ``blocks.3.hook_resid_post``.
    """
    name = TRANSFORMER_ACT_NAME[activation]
    return name if layer is None else f"blocks.{layer}.{name}"


def check_transformer_activations(model: Model, names: Iterable[str]) -> None:
    """Check that the model computes every transformer activation a run reads or writes.

    Which it computes depends on its backend and architecture: a ``HookedTransformer`` has none of a
    ``TransformerBridge``'s sublayer inputs and outputs, a model without a gated MLP no ``mlp.pre_linear`` and a
    parallel block no ``resid.mid``. A row of a block's activation that names several layers, or none, names one
    without its block, which no model computes. An input checks its ``activation_names()`` as it resolves its
    index, so such a run is refused before any forward pass, and by its plan too, rather than failing in
    ``transformer_lens`` once it looks the activation up.

    Args:
        model: The model the run reads.
        names: The transformer activation names the run reads or writes.

    Raises:
        ValueError: If the model computes no transformer activation of one of the names.
    """
    if missing := sorted(set(names) - model.hook_dict.keys()):
        raise ValueError(
            f"This model has no transformer activation {', '.join(map(repr, missing))}: its architecture or its "
            "backend does not compute it, or a row reads a block's activation without a single layer. A "
            "HookedTransformer has none of a TransformerBridge's sublayer inputs and outputs"
        )


SUBLAYER_INPUTS = frozenset(
    TRANSFORMER_ACT_NAME[activation] for activation in (ModelLayerLocation.ATTENTION_IN, ModelLayerLocation.MLP_IN)
)


def writable(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Pick the tensor a hook writes into and hands on.

    Any other activation is written in place, so the write reaches whatever else holds the tensor, which ``resid.mid``
    needs: a bridge hooks it on the input of the norm before the MLP, and only an in-place write reaches the
    residual stream the block keeps beside that norm. The name is the one the hook is registered under, as
    ``transformer_activation`` gives it, not the hook point's own: OLMo 2's ``hook_resid_mid`` is ``mlp.hook_in``
    by another name.

    Args:
        name: The transformer activation name the hook writes.
        tensor: The tensor the hook is handed.

    Returns:
        A copy of ``tensor`` for a sublayer's input, which only the sublayer reads, and ``tensor`` itself otherwise.
    """
    return tensor.clone() if name.split(".", 2)[-1] in SUBLAYER_INPUTS else tensor
