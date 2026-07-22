from collections.abc import Sequence

import torch
from jaxtyping import Float

from intex.lens.method.lens import AbstractLensMethod
from intex.model.config import ModelLocation, ModelLocatorConfig
from intex.model.index import IndexElement


class LogitLens(AbstractLensMethod):
    def project(
        self,
        activation: Float[torch.Tensor, "..."],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "... d_vocab"]:
        with torch.no_grad():
            residual = self._to_residual(activation, index)
            return self.model.ln_final(residual) @ self.model.W_U + self.model.b_U

    def _to_residual(
        self,
        activation: Float[torch.Tensor, "..."],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "... d_model"]:
        return activation


class MLPLogitLens(LogitLens):
    def _to_residual(
        self,
        activation: Float[torch.Tensor, "..."],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "... d_model"]:
        layer = index.layer()
        if layer is None:
            raise ValueError(
                f"Projecting '{index.activation().value}' needs a single layer: 'W_out' is per layer, so the "
                "element cannot leave 'layer' out or group several of them"
            )
        w_out = self.model.blocks[layer].mlp.W_out  # [d_mlp, d_model]

        neuron = index.get(ModelLocation.NEURON)
        if neuron is None:
            return activation @ w_out
        if isinstance(neuron, Sequence):
            return activation @ w_out[list(neuron)]
        return activation.unsqueeze(-1) * w_out[neuron]
