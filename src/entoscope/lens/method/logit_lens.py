import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from jaxtyping import Float

from entoscope.lens.method.lens import AbstractLensMethod, LensMethodConfigT
from entoscope.model import Model
from entoscope.model.config import ModelLayerLocation, ModelLocation, ModelLocatorConfig
from entoscope.model.index import IndexElement
from entoscope.model.weight import attention_out_weight, mlp_weight

if TYPE_CHECKING:  # ``entoscope.lens.input`` looks the lens up through the factory, which imports this module.
    from entoscope.lens.input import LensInput

_KEY_VALUE_ACTIVATIONS = (ModelLayerLocation.ATTENTION_K, ModelLayerLocation.ATTENTION_V)
_QUERY_KEY_ACTIVATIONS = (ModelLayerLocation.ATTENTION_Q, ModelLayerLocation.ATTENTION_K)


def _attention_scale(model: Model) -> float:
    # ``TransformerBridgeConfig`` leaves ``attn_scale`` at the -1.0 sentinel that
    # ``HookedTransformerConfig`` resolves for itself in ``__post_init__``.
    if not model.cfg.use_attn_scale:
        return 1.0
    scale = model.cfg.attn_scale
    return scale if scale and scale > 0 else math.sqrt(model.cfg.d_head)


def _layer(index: IndexElement[ModelLocatorConfig]) -> int:
    layer = index.layer()
    if layer is None:
        raise ValueError(
            f"Projecting '{index.activation().value}' needs a single layer: the projection weight is per layer, "
            "so the element cannot leave 'layer' out or group several of them"
        )
    return layer


def _axis_index(index: IndexElement[ModelLocatorConfig], location: ModelLocation) -> int | list[int] | slice:
    # An int drops the axis, the same way ``IndexElement.slices()`` drops it on the activation, so
    # ``_head_lift`` contracts the same axes on both sides.
    value = index.get(location)
    if value is None:
        return slice(None)
    return list(value) if isinstance(value, Sequence) else value


def _attention_layer(model: Model, index: IndexElement[ModelLocatorConfig]) -> int:
    if index.activation() in _KEY_VALUE_ACTIVATIONS and model.cfg.n_key_value_heads not in (None, model.cfg.n_heads):
        raise ValueError(
            f"Projecting '{index.activation().value}' is not supported on grouped-query attention: it has "
            f"{model.cfg.n_key_value_heads} heads while the projection weight has {model.cfg.n_heads}"
        )
    return _layer(index)


def _head_lift(
    activation: Float[torch.Tensor, "..."],
    weight: Float[torch.Tensor, "head d_head d_out"],
    index: IndexElement[ModelLocatorConfig],
) -> Float[torch.Tensor, "... d_out"]:
    weight = weight[_axis_index(index, ModelLocation.HEAD), _axis_index(index, ModelLocation.NEURON)]
    return torch.tensordot(activation, weight, dims=weight.ndim - 1)


class LogitLens(AbstractLensMethod[LensMethodConfigT]):
    """Reads an element out the way the model reads out its final residual stream.

    A residual-stream state is unembedded as it is. An MLP-hidden or attention head activation is
    first written into the residual stream through the weight that writes it there, and a query or
    key is read against the embedding through the other side of its head.
    """

    def extract(
        self,
        lens_input: "LensInput",
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "..."]:
        return lens_input.cache[index.transformer_activation()][index.slices()]

    def project(
        self,
        activation: Float[torch.Tensor, "..."],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "... d_vocab"]:
        with torch.no_grad():
            if index.activation() in _QUERY_KEY_ACTIVATIONS:
                return self._query_key_logits(activation, index)
            residual = self._to_final_residual(self._to_residual(activation, index), index)
            return self.model.ln_final(residual) @ self.model.W_U + self.model.b_U

    def _to_residual(
        self,
        activation: Float[torch.Tensor, "..."],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "... d_model"]:
        match index.activation():
            case (
                ModelLayerLocation.EMBEDDING
                | ModelLayerLocation.RESIDUAL_PRE
                | ModelLayerLocation.RESIDUAL_MID
                | ModelLayerLocation.RESIDUAL_POST
                | ModelLayerLocation.ATTENTION_OUT
                | ModelLayerLocation.MLP_OUT
            ):
                return activation
            case ModelLayerLocation.MLP_PRE | ModelLayerLocation.MLP_PRE_LINEAR | ModelLayerLocation.MLP_POST:
                return self._mlp_to_residual(activation, index)
            case ModelLayerLocation.ATTENTION_V | ModelLayerLocation.ATTENTION_Z:
                layer = _attention_layer(self.model, index)
                return _head_lift(activation, attention_out_weight(self.model, layer), index)
            case unprojectable:
                raise ValueError(
                    f"{type(self).__name__} cannot project {unprojectable.value}: only residual-stream, MLP-hidden "
                    "and attention head activations map onto the vocabulary"
                )

    def _to_final_residual(
        self,
        residual: Float[torch.Tensor, "... d_model"],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "... d_model"]:
        """Where a lens moves the residual stream before the model's own read-out (the tuned lens translates it)."""
        return residual

    def _mlp_to_residual(
        self,
        activation: Float[torch.Tensor, "..."],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "... d_model"]:
        layer = _layer(index)
        w_out = mlp_weight(self.model, layer, "W_out")  # [d_mlp, d_model]

        neuron = index.get(ModelLocation.NEURON)
        if neuron is None:
            return activation @ w_out
        if isinstance(neuron, Sequence):
            return activation @ w_out[list(neuron)]
        return activation.unsqueeze(-1) * w_out[neuron]

    def _query_key_logits(
        self,
        activation: Float[torch.Tensor, "..."],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "... d_vocab"]:
        layer = _attention_layer(self.model, index)
        attention = self.model.blocks[layer].attn
        weight = attention.W_K if index.activation() is ModelLayerLocation.ATTENTION_Q else attention.W_Q
        embedding = _head_lift(activation, weight.transpose(-1, -2), index)
        return embedding @ self.model.W_E.T / _attention_scale(self.model)
