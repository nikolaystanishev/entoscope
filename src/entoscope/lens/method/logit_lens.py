import math
from typing import TYPE_CHECKING

import torch
from jaxtyping import Float

from entoscope.lens.method.lens import AbstractLensMethod, LensMethodConfigT
from entoscope.model import Model
from entoscope.model.config import ModelLayerLocation, ModelLocatorConfig
from entoscope.model.index import IndexElement
from entoscope.model.weight import ModelWeight, activation_weight, model_weight

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
_WRITTEN_AS = {
    ModelLayerLocation.ATTENTION_V: ModelLayerLocation.ATTENTION_Z,
    ModelLayerLocation.ATTENTION_Z: ModelLayerLocation.ATTENTION_Z,
    ModelLayerLocation.MLP_PRE: ModelLayerLocation.MLP_POST,
    ModelLayerLocation.MLP_PRE_LINEAR: ModelLayerLocation.MLP_POST,
    ModelLayerLocation.MLP_POST: ModelLayerLocation.MLP_POST,
}
_QUERY_KEY_PARTNER = {
    ModelLayerLocation.ATTENTION_Q: ModelLayerLocation.ATTENTION_K,
    ModelLayerLocation.ATTENTION_K: ModelLayerLocation.ATTENTION_Q,
}


def projection_layer(model: Model, index: IndexElement[ModelLocatorConfig]) -> int:
    """Find the single layer whose weights project an element.

    Args:
        model: The model the element is read from.
        index: The element.

    Returns:
        The element's layer.

    Raises:
        ValueError: If the element names no single layer, or is a key or a value of grouped-query
            attention, which has fewer heads than the projection weights.
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
    """Carry an activation through a weight that holds one ``d_model`` vector per unit of the activation.

    The weight is sliced at the units the element names as the activation was, by the activation's axes
    after ``batch`` and ``position``, so the two keep the same axes and contract over all of them.

    Args:
        activation: The element's activation.
        weight: The weight, as ``model_weight`` returns it.
        index: The element.

    Returns:
        The activation's contribution in the residual stream's basis.
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
        """Read the element's activation out of the prompt's cache.

        Args:
            lens_input: The prepared input.
            index: The element.

        Returns:
            The activation, sliced at the element.
        """
        return lens_input.cache[index.transformer_activation()][index.slices()]

    @torch.no_grad()
    def project(
        self,
        activation: Float[torch.Tensor, "..."],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "... d_vocab"]:
        """Read an activation out through the model's final norm and unembedding.

        Args:
            activation: The element's activation.
            index: The element.

        Returns:
            The element's logits over the vocabulary.

        Raises:
            ValueError: If the activation is neither in the residual stream nor an MLP-hidden or attention
                head activation, or names no single layer where it needs one.
        """
        if index.activation() in _QUERY_KEY_PARTNER:
            return self._query_key_logits(activation, index)
        residual = self._to_final_residual(self._to_residual(activation, index), index)
        return self.model.ln_final(residual) @ model_weight(self.model, ModelWeight.UNEMBED).T + self.model.b_U

    def _to_residual(
        self,
        activation: Float[torch.Tensor, "..."],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "... d_model"]:
        if index.activation() in _RESIDUAL_STREAM:
            return activation
        if index.activation() in _WRITTEN_AS:
            weight = activation_weight(self.model, _WRITTEN_AS[index.activation()], projection_layer(self.model, index))
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
        """Move the residual stream before the model's own read-out, which the tuned lens overrides to translate it.

        Args:
            residual: The element's contribution to the residual stream.
            index: The element.

        Returns:
            The residual stream to read out, unchanged here.
        """
        return residual

    def _query_key_logits(
        self,
        activation: Float[torch.Tensor, "..."],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "... d_vocab"]:
        partner = activation_weight(
            self.model, _QUERY_KEY_PARTNER[index.activation()], projection_layer(self.model, index)
        )
        embedding = model_weight(self.model, ModelWeight.EMBED)
        return _through(activation, partner, index) @ embedding.T / _attention_scale(self.model)
