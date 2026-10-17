import einops
import torch
from jaxtyping import Float

from entoscope.model.base import Model, is_bridge
from entoscope.model.config import ModelLayerLocation

# The bridge submodule holding each HookedTransformer-style MLP weight.
_MLP_SUBMODULE = {"W_in": "in", "W_gate": "gate", "W_out": "out"}

READ_WEIGHT: dict[ModelLayerLocation, str] = {
    ModelLayerLocation.MLP_PRE: "W_gate",
    ModelLayerLocation.MLP_PRE_LINEAR: "W_in",
    ModelLayerLocation.ATTENTION_Q: "W_Q",
    ModelLayerLocation.ATTENTION_K: "W_K",
    ModelLayerLocation.ATTENTION_V: "W_V",
}

WRITE_WEIGHT: dict[ModelLayerLocation, str] = {
    ModelLayerLocation.MLP_PRE: "W_out",
    ModelLayerLocation.MLP_PRE_LINEAR: "W_out",
    ModelLayerLocation.MLP_POST: "W_out",
    ModelLayerLocation.ATTENTION_V: "W_O",
    ModelLayerLocation.ATTENTION_Z: "W_O",
}


def read_weight(model: Model, layer: int, activation: ModelLayerLocation) -> Float[torch.Tensor, "... d_model"]:
    """The weight ``READ_WEIGHT`` names for ``activation``: what computes it off the residual stream.

    One ``d_model`` vector per unit of the activation, ``[neuron, d_model]`` for an MLP activation and
    ``[head, d_head, d_model]`` for an attention one, as ``write_weight`` returns its own. A gated MLP computes
    ``mlp.pre`` with its gate and ``mlp.pre_linear`` with ``W_in``; any other MLP computes ``mlp.pre`` with
    ``W_in`` and has no ``mlp.pre_linear``.
    """
    name = READ_WEIGHT[activation]
    if not model.cfg.gated_mlp:
        if activation is ModelLayerLocation.MLP_PRE_LINEAR:
            raise ValueError(
                "'mlp.pre_linear' is the linear branch of a gated MLP, which this model does not have; "
                "read its input weights through 'mlp.pre'"
            )
        if activation is ModelLayerLocation.MLP_PRE:
            name = "W_in"
    return _weight(model, layer, name).transpose(-1, -2)


def write_weight(model: Model, layer: int, activation: ModelLayerLocation) -> Float[torch.Tensor, "... d_model"]:
    """The weight ``WRITE_WEIGHT`` names for ``activation``: what carries it into the residual stream.

    One ``d_model`` vector per unit of the activation, as ``read_weight`` returns its own.
    """
    return _weight(model, layer, WRITE_WEIGHT[activation])


def _weight(model: Model, layer: int, name: str) -> torch.Tensor:
    if name in _MLP_SUBMODULE:
        return mlp_weight(model, layer, name)
    if name == "W_O":
        return attention_out_weight(model, layer)
    return getattr(model.blocks[layer].attn, name)


def mlp_weight(model: Model, layer: int, attribute: str) -> torch.Tensor | None:
    """A block's MLP weight in HookedTransformer's orientation.

    ``W_in`` and ``W_gate`` come back as ``[d_model, d_mlp]`` and ``W_out`` as ``[d_mlp, d_model]``.
    On a bridge the weight is read off the wrapped module's parameter, whose layout its type fixes:
    ``nn.Linear`` stores ``[out, in]`` while GPT-2's ``Conv1D`` stores ``[in, out]``, so only the
    former needs transposing. The bridge's own ``W_in``-style aliases are no help: transformer-lens has
    returned an ``nn.Linear`` weight through them in either orientation, depending on the release.
    ``enable_compatibility_mode`` processes the parameter in place, so the processed weights come back.
    """
    mlp = model.blocks[layer].mlp
    if not is_bridge(model):
        return getattr(mlp, attribute)

    submodule = mlp._modules.get(_MLP_SUBMODULE[attribute])
    if submodule is None:  # ``W_gate`` of an MLP without a gate
        return None
    component = submodule.original_component
    return component.weight.T if isinstance(component, torch.nn.Linear) else component.weight


def attention_out_weight(model: Model, layer: int) -> torch.Tensor:
    """A block's ``W_O`` in HookedTransformer's orientation, ``[head, d_head, d_model]``.

    A bridge picks its reshape by reading the layout off the matrix shape, and ``W_O`` is square
    whenever ``d_model == n_heads * d_head`` - so on the architectures ``mlp_weight`` describes it
    takes the ``Conv1D`` branch and skips the transpose ``nn.Linear`` needs. Rebuilding from the
    parameter also covers the non-square case, where the bridge's reading is already right.
    """
    attn = model.blocks[layer].attn
    out = attn._modules.get("o") if is_bridge(model) else None
    if out is None or not isinstance(out.original_component, torch.nn.Linear):
        return attn.W_O
    return einops.rearrange(out.weight.T, "(head d_head) d_model -> head d_head d_model", head=model.cfg.n_heads)
