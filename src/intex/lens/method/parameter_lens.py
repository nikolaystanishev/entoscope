import abc
from collections.abc import Sequence

import torch
from jaxtyping import Float

from intex.lens.config import ProjectionSpace
from intex.lens.input import LensInput
from intex.lens.method.lens import AbstractLensMethod
from intex.lens.method.logit_lens import _attention_layer, _layer
from intex.model.config import ModelLayerLocation, ModelLocation, ModelLocatorConfig
from intex.model.index import IndexElement

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


class AbstractParameterLensMethod(AbstractLensMethod):
    def project(
        self,
        weight: Float[torch.Tensor, " d_model"],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, " d_vocab"]:
        with torch.no_grad():
            return (self.config.scale * weight) @ self._projection()

    def _projection(self) -> Float[torch.Tensor, "d_model d_vocab"]:
        if (self.config.projection or ProjectionSpace.UNEMBEDDING) is ProjectionSpace.EMBEDDING:
            return self.model.W_E.T
        return self.model.W_U

    @abc.abstractmethod
    def extract(
        self,
        lens_input: LensInput,
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, " d_model"]:
        raise NotImplementedError


class MLPKeyParameterLens(AbstractParameterLensMethod):
    def extract(
        self,
        lens_input: LensInput,
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, " d_model"]:
        gated = bool(self.model.cfg.gated_mlp)
        if index.activation() is ModelLayerLocation.MLP_PRE_LINEAR and not gated:
            raise ValueError(
                "'mlp.pre_linear' is the linear branch of a gated MLP, which this model does not have; "
                "read its input weights through 'mlp.pre'"
            )
        layer, neuron = _layer(index), _selection(index, ModelLocation.NEURON)

        mlp = self.model.blocks[layer].mlp
        weight = mlp.W_gate if gated and index.activation() is ModelLayerLocation.MLP_PRE else mlp.W_in
        return weight[:, neuron].sum(-1)


class MLPValueParameterLens(AbstractParameterLensMethod):
    def extract(
        self,
        lens_input: LensInput,
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, " d_model"]:
        layer, neuron = _layer(index), _selection(index, ModelLocation.NEURON)
        return self.model.blocks[layer].mlp.W_out[neuron].sum(0)


class AttentionReadParameterLens(AbstractParameterLensMethod):
    def extract(
        self,
        lens_input: LensInput,
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, " d_model"]:
        layer = _attention_layer(self.model, index)
        head, neuron = _selection(index, ModelLocation.HEAD), _selection(index, ModelLocation.NEURON)

        weight = getattr(self.model.blocks[layer].attn, _ATTENTION_READ_WEIGHT[index.activation()])
        return weight[head][..., neuron].sum((0, -1))


class AttentionWriteParameterLens(AbstractParameterLensMethod):
    def extract(
        self,
        lens_input: LensInput,
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, " d_model"]:
        layer = _attention_layer(self.model, index)
        head, neuron = _selection(index, ModelLocation.HEAD), _selection(index, ModelLocation.NEURON)

        return self.model.blocks[layer].attn.W_O[head][:, neuron].sum((0, 1))
