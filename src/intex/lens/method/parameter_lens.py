from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch
from jaxtyping import Float

from intex.lens.config import LensSettings
from intex.lens.method.lens import AbstractLensMethod
from intex.lens.method.logit_lens import _attention_layer, _layer
from intex.model.config import ModelLayerLocation, ModelLocation, ModelLocatorConfig
from intex.model.index import IndexElement
from intex.model.weight import attention_out_weight, mlp_weight

if TYPE_CHECKING:  # ``intex.lens.input`` looks the lens up through the factory, which imports this module.
    from intex.lens.input import LensInput

_ATTENTION_READ_WEIGHT = {
    ModelLayerLocation.ATTENTION_Q: "W_Q",
    ModelLayerLocation.ATTENTION_K: "W_K",
    ModelLayerLocation.ATTENTION_V: "W_V",
}


def _selection(index: IndexElement[ModelLocatorConfig], location: ModelLocation) -> list[int] | slice:
    # Always a list or a slice: a single advanced index per bracket keeps the axis and never pairs
    # 'head' with 'neuron' the way two list indices in one bracket would.
    value = index.get(location)
    if value is None:
        return slice(None)
    return list(value) if isinstance(value, Sequence) else [value]


class ProjectionSpace(str, Enum):
    EMBEDDING = "embedding"
    UNEMBEDDING = "unembedding"


@dataclass(kw_only=True)
class ParameterLensSettings(LensSettings):
    """``projection`` picks the vocabulary space a weight is read in, and ``scale`` multiplies the weight
    first: a negative scale reads it against its own direction, and the magnitude acts as a temperature."""

    projection: ProjectionSpace = ProjectionSpace.UNEMBEDDING
    scale: float = 1.0


class ParameterLens(AbstractLensMethod):
    """Reads a weight vector onto the vocabulary: what a neuron or a head dimension reads or writes.

    It reads the model's weights rather than a prompt's activations, so a run takes no prompt.
    """

    reads_activations = False
    settings_class = ParameterLensSettings

    def extract(
        self,
        lens_input: "LensInput",
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, " d_model"]:
        match index.activation():
            case ModelLayerLocation.MLP_PRE | ModelLayerLocation.MLP_PRE_LINEAR:
                return self._mlp_key(index)
            case ModelLayerLocation.MLP_POST:
                return self._mlp_value(index)
            case ModelLayerLocation.ATTENTION_Q | ModelLayerLocation.ATTENTION_K | ModelLayerLocation.ATTENTION_V:
                return self._attention_read(index)
            case ModelLayerLocation.ATTENTION_Z:
                return self._attention_write(index)
            case unprojectable:
                raise ValueError(
                    f"{type(self).__name__} cannot project {unprojectable.value}: only MLP-hidden and attention head "
                    "locations name a weight vector, one per neuron or head dimension. Residual-stream locations "
                    "('embed', 'resid.*', 'attn.out', 'mlp.out') index the residual basis instead of a weight"
                )

    def project(
        self,
        weight: Float[torch.Tensor, " d_model"],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, " d_vocab"]:
        with torch.no_grad():
            return (self.settings.scale * weight) @ self._projection()

    def _projection(self) -> Float[torch.Tensor, "d_model d_vocab"]:
        if self.settings.projection is ProjectionSpace.EMBEDDING:
            return self.model.W_E.T
        return self.model.W_U

    def _mlp_key(self, index: IndexElement[ModelLocatorConfig]) -> Float[torch.Tensor, " d_model"]:
        gated = bool(self.model.cfg.gated_mlp)
        if index.activation() is ModelLayerLocation.MLP_PRE_LINEAR and not gated:
            raise ValueError(
                "'mlp.pre_linear' is the linear branch of a gated MLP, which this model does not have; "
                "read its input weights through 'mlp.pre'"
            )
        layer, neuron = _layer(index), _selection(index, ModelLocation.NEURON)

        model_attribute = "W_gate" if gated and index.activation() is ModelLayerLocation.MLP_PRE else "W_in"
        return mlp_weight(self.model, layer, model_attribute)[:, neuron].sum(-1)

    def _mlp_value(self, index: IndexElement[ModelLocatorConfig]) -> Float[torch.Tensor, " d_model"]:
        layer, neuron = _layer(index), _selection(index, ModelLocation.NEURON)
        return mlp_weight(self.model, layer, "W_out")[neuron].sum(0)

    def _attention_read(self, index: IndexElement[ModelLocatorConfig]) -> Float[torch.Tensor, " d_model"]:
        layer = _attention_layer(self.model, index)
        head, neuron = _selection(index, ModelLocation.HEAD), _selection(index, ModelLocation.NEURON)

        weight = getattr(self.model.blocks[layer].attn, _ATTENTION_READ_WEIGHT[index.activation()])
        return weight[head][..., neuron].sum((0, -1))

    def _attention_write(self, index: IndexElement[ModelLocatorConfig]) -> Float[torch.Tensor, " d_model"]:
        layer = _attention_layer(self.model, index)
        head, neuron = _selection(index, ModelLocation.HEAD), _selection(index, ModelLocation.NEURON)

        return attention_out_weight(self.model, layer)[head][:, neuron].sum((0, 1))
