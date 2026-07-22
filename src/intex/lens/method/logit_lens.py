from collections.abc import Sequence

import torch
from jaxtyping import Float

from intex.lens.input import LensInput
from intex.lens.method.lens import AbstractLensMethod
from intex.model import Model
from intex.model.config import ModelLayerLocation, ModelLocation, ModelLocatorConfig
from intex.model.index import IndexElement

_KEY_VALUE_ACTIVATIONS = (ModelLayerLocation.ATTENTION_K, ModelLayerLocation.ATTENTION_V)


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


class LogitLens(AbstractLensMethod):
    def extract(
        self,
        lens_input: LensInput,
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "..."]:
        return lens_input.cache[index.transformer_activation()][index.slices()]

    def project(
        self,
        activation: Float[torch.Tensor, "..."],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "... d_vocab"]:
        with torch.no_grad():
            residual = self._to_final_residual(self._to_residual(activation, index), index)
            return self.model.ln_final(residual) @ self.model.W_U + self.model.b_U

    def _to_residual(
        self,
        activation: Float[torch.Tensor, "..."],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "... d_model"]:
        return activation

    def _to_final_residual(
        self,
        residual: Float[torch.Tensor, "... d_model"],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "... d_model"]:
        return residual


class MLPLogitLens(LogitLens):
    def _to_residual(
        self,
        activation: Float[torch.Tensor, "..."],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "... d_model"]:
        layer = _layer(index)
        w_out = self.model.blocks[layer].mlp.W_out  # [d_mlp, d_model]

        neuron = index.get(ModelLocation.NEURON)
        if neuron is None:
            return activation @ w_out
        if isinstance(neuron, Sequence):
            return activation @ w_out[list(neuron)]
        return activation.unsqueeze(-1) * w_out[neuron]


class AttentionLogitLens(LogitLens):
    def _to_residual(
        self,
        activation: Float[torch.Tensor, "..."],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "... d_model"]:
        layer = _attention_layer(self.model, index)
        return _head_lift(activation, self.model.blocks[layer].attn.W_O, index)


class AttentionQKLogitLens(LogitLens):
    def project(
        self,
        activation: Float[torch.Tensor, "..."],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "... d_vocab"]:
        with torch.no_grad():
            layer = _attention_layer(self.model, index)
            attention = self.model.blocks[layer].attn
            weight = attention.W_K if index.activation() is ModelLayerLocation.ATTENTION_Q else attention.W_Q
            embedding = _head_lift(activation, weight.transpose(-1, -2), index)
            return embedding @ self.model.W_E.T / self.model.cfg.attn_scale
