from enum import StrEnum

import einops
import torch
from jaxtyping import Float

from entoscope.model.base import Model, is_bridge
from entoscope.model.config import ModelLayerLocation


class ModelWeight(StrEnum):
    """A weight of the model, by its ``HookedTransformer`` name."""

    EMBED = "W_E"
    UNEMBED = "W_U"
    QUERY = "W_Q"
    KEY = "W_K"
    VALUE = "W_V"
    ATTENTION_OUT = "W_O"
    MLP_IN = "W_in"
    MLP_GATE = "W_gate"
    MLP_OUT = "W_out"


_MLP_SUBMODULE = {ModelWeight.MLP_IN: "in", ModelWeight.MLP_GATE: "gate", ModelWeight.MLP_OUT: "out"}

ACTIVATION_WEIGHT: dict[ModelLayerLocation, ModelWeight] = {
    ModelLayerLocation.ATTENTION_Q: ModelWeight.QUERY,
    ModelLayerLocation.ATTENTION_K: ModelWeight.KEY,
    ModelLayerLocation.ATTENTION_V: ModelWeight.VALUE,
    ModelLayerLocation.ATTENTION_Z: ModelWeight.ATTENTION_OUT,
    ModelLayerLocation.MLP_PRE: ModelWeight.MLP_GATE,
    ModelLayerLocation.MLP_PRE_LINEAR: ModelWeight.MLP_IN,
    ModelLayerLocation.MLP_POST: ModelWeight.MLP_OUT,
}


def activation_weight(model: Model, activation: ModelLayerLocation, layer: int) -> Float[torch.Tensor, "... d_model"]:
    """The weight ``ACTIVATION_WEIGHT`` names for ``activation``, as ``model_weight`` returns it.

    That is the weight computing the activation off the residual stream, or, for one other activations
    compute (``attn.z``, ``mlp.post``), the weight writing it back. A gated MLP computes ``mlp.pre`` with its
    gate and ``mlp.pre_linear`` with ``W_in``; any other MLP computes ``mlp.pre`` with ``W_in`` and has no
    ``mlp.pre_linear``.
    """
    name = ACTIVATION_WEIGHT[activation]
    if not model.cfg.gated_mlp:
        if activation is ModelLayerLocation.MLP_PRE_LINEAR:
            raise ValueError(
                "'mlp.pre_linear' is the linear branch of a gated MLP, which this model does not have; "
                "read its input weights through 'mlp.pre'"
            )
        if name is ModelWeight.MLP_GATE:
            name = ModelWeight.MLP_IN
    return model_weight(model, name, layer)


def model_weight(model: Model, name: ModelWeight, layer: int | None = None) -> Float[torch.Tensor, "... d_model"]:
    """``name`` as one ``d_model`` vector per unit it reads or writes, whichever side of the residual stream it is on.

    ``[d_vocab, d_model]`` for the embedding and unembedding, ``[d_mlp, d_model]`` for an MLP weight and
    ``[head, d_head, d_model]`` for an attention one. ``layer`` names the block of every weight but those two.
    """
    match name:
        case ModelWeight.EMBED:
            return model.W_E
        case ModelWeight.UNEMBED:
            return model.W_U.T
        case ModelWeight.ATTENTION_OUT:
            return _attention_out_weight(model, _block(name, layer))
        case ModelWeight.MLP_OUT:
            return _mlp_weight(model, _block(name, layer), name)
        case ModelWeight.MLP_IN | ModelWeight.MLP_GATE:
            return _mlp_weight(model, _block(name, layer), name).T
        case ModelWeight.QUERY | ModelWeight.KEY | ModelWeight.VALUE:
            return getattr(model.blocks[_block(name, layer)].attn, name).transpose(-1, -2)


def _block(name: ModelWeight, layer: int | None) -> int:
    if layer is None:
        raise ValueError(f"'{name}' is a weight of every block, so reading it needs a layer")
    return layer


def _mlp_weight(model: Model, layer: int, name: ModelWeight) -> torch.Tensor:
    """A block's MLP weight in HookedTransformer's orientation.

    ``W_in`` and ``W_gate`` come back as ``[d_model, d_mlp]`` and ``W_out`` as ``[d_mlp, d_model]``.
    On a bridge the weight is read off the wrapped module's parameter, whose layout its type fixes:
    ``nn.Linear`` stores ``[out, in]`` while GPT-2's ``Conv1D`` stores ``[in, out]``, so only the
    former needs transposing. The bridge's own ``W_in``-style aliases are no help: transformer-lens has
    returned an ``nn.Linear`` weight through them in either orientation, depending on the release.
    ``enable_compatibility_mode`` processes the parameter in place, so the processed weights come back.
    """
    if name is ModelWeight.MLP_GATE and not model.cfg.gated_mlp:
        raise ValueError("This model's MLP has no gate, so it has no 'W_gate'")

    mlp = model.blocks[layer].mlp
    if not is_bridge(model):
        return getattr(mlp, name)

    component = mlp._modules[_MLP_SUBMODULE[name]].original_component
    return component.weight.T if isinstance(component, torch.nn.Linear) else component.weight


def _attention_out_weight(model: Model, layer: int) -> torch.Tensor:
    """A block's ``W_O`` in HookedTransformer's orientation, ``[head, d_head, d_model]``.

    A bridge picks its reshape by reading the layout off the matrix shape, and ``W_O`` is square
    whenever ``d_model == n_heads * d_head`` - so on the architectures ``_mlp_weight`` describes it
    takes the ``Conv1D`` branch and skips the transpose ``nn.Linear`` needs. Rebuilding from the
    parameter also covers the non-square case, where the bridge's reading is already right.
    """
    attn = model.blocks[layer].attn
    out = attn._modules.get("o") if is_bridge(model) else None
    if out is None or not isinstance(out.original_component, torch.nn.Linear):
        return attn.W_O
    return einops.rearrange(out.weight.T, "(head d_head) d_model -> head d_head d_model", head=model.cfg.n_heads)
