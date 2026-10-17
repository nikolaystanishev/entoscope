import math
from typing import TYPE_CHECKING

import torch
from jaxtyping import Float

from entoscope.lens.method.lens import AbstractLensMethod, LensMethodConfigT
from entoscope.model import Model
from entoscope.model.config import ModelLayerLocation, ModelLocatorConfig
from entoscope.model.index import IndexElement
from entoscope.model.weight import WRITE_WEIGHT, read_weight, write_weight

if TYPE_CHECKING:
    from entoscope.lens.input import LensInput

_RESIDUAL_STREAM = (
    ModelLayerLocation.EMBEDDING,
    ModelLayerLocation.RESIDUAL_PRE,
    ModelLayerLocation.RESIDUAL_MID,
    ModelLayerLocation.RESIDUAL_POST,
    ModelLayerLocation.ATTENTION_OUT,
    ModelLayerLocation.MLP_OUT,
)
_QUERY_KEY_PARTNER = {
    ModelLayerLocation.ATTENTION_Q: ModelLayerLocation.ATTENTION_K,
    ModelLayerLocation.ATTENTION_K: ModelLayerLocation.ATTENTION_Q,
}


def projection_layer(model: Model, index: IndexElement[ModelLocatorConfig]) -> int:
    """The element's single layer, whose weights project it.

    Raises for a key or a value of grouped-query attention, which has fewer heads than those weights.
    """
    layer = index.layer()
    if layer is None:
        raise ValueError(
            f"Projecting '{index.activation().value}' needs a single layer: the projection weight is per layer, "
            "so the element cannot leave 'layer' out or group several of them"
        )
    if index.activation() in (ModelLayerLocation.ATTENTION_K, ModelLayerLocation.ATTENTION_V) and (
        model.cfg.n_key_value_heads not in (None, model.cfg.n_heads)
    ):
        raise ValueError(
            f"Projecting '{index.activation().value}' is not supported on grouped-query attention: it has "
            f"{model.cfg.n_key_value_heads} heads while the projection weight has {model.cfg.n_heads}"
        )
    return layer


def _attention_scale(model: Model) -> float:
    if not model.cfg.use_attn_scale:
        return 1.0
    # ``TransformerBridgeConfig`` leaves ``attn_scale`` at the -1.0 that ``HookedTransformerConfig`` resolves.
    scale = model.cfg.attn_scale
    return scale if scale and scale > 0 else math.sqrt(model.cfg.d_head)


def _through(
    activation: Float[torch.Tensor, "..."],
    weight: Float[torch.Tensor, "... d_model"],
    index: IndexElement[ModelLocatorConfig],
) -> Float[torch.Tensor, "... d_model"]:
    """``activation`` carried through ``weight``, one ``d_model`` vector per unit of the activation.

    The weight is sliced at the units the element names as the activation was, by the activation's axes after
    ``batch`` and ``position``, so the two keep the same axes and contract over all of them.
    """
    weight = weight[index.slices()[2:]]
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

    @torch.no_grad()
    def project(
        self,
        activation: Float[torch.Tensor, "..."],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "... d_vocab"]:
        if index.activation() in _QUERY_KEY_PARTNER:
            return self._query_key_logits(activation, index)
        residual = self._to_final_residual(self._to_residual(activation, index), index)
        return self.model.ln_final(residual) @ self.model.W_U + self.model.b_U

    def _to_residual(
        self,
        activation: Float[torch.Tensor, "..."],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "... d_model"]:
        if index.activation() in _RESIDUAL_STREAM:
            return activation
        if index.activation() in WRITE_WEIGHT:
            weight = write_weight(self.model, projection_layer(self.model, index), index.activation())
            return _through(activation, weight, index)
        raise ValueError(
            f"{type(self).__name__} cannot project {index.activation().value}: only residual-stream, MLP-hidden "
            "and attention head activations map onto the vocabulary"
        )

    def _to_final_residual(
        self,
        residual: Float[torch.Tensor, "... d_model"],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "... d_model"]:
        """Where a lens moves the residual stream before the model's own read-out (the tuned lens translates it)."""
        return residual

    def _query_key_logits(
        self,
        activation: Float[torch.Tensor, "..."],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "... d_vocab"]:
        partner = read_weight(self.model, projection_layer(self.model, index), _QUERY_KEY_PARTNER[index.activation()])
        return _through(activation, partner, index) @ self.model.W_E.T / _attention_scale(self.model)
