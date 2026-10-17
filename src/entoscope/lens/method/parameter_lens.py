from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import torch
from jaxtyping import Float

from entoscope.lens.config import LensMethodConfig
from entoscope.lens.method.lens import AbstractLensMethod
from entoscope.lens.method.logit_lens import projection_layer
from entoscope.model.config import ModelLocation, ModelLocatorConfig
from entoscope.model.index import IndexElement
from entoscope.model.slicer import ActivationSlicer
from entoscope.model.weight import ACTIVATION_WEIGHT, ModelWeight, activation_weight, model_weight

if TYPE_CHECKING:
    from entoscope.lens.input import LensInput


def _selection(index: IndexElement[ModelLocatorConfig], location: ModelLocation) -> list[int] | slice:
    # Always a list or a slice: a single advanced index per bracket keeps the axis and never pairs
    # 'head' with 'neuron' the way two list indices in one bracket would.
    value = index.get(location)
    if value is None:
        return slice(None)
    return list(value) if isinstance(value, Sequence) else [value]


class ProjectionSpace(StrEnum):
    """The vocabulary space a weight is read in: ``W_U`` for ``unembedding``, ``W_E`` transposed for ``embedding``.

    The two coincide on a model that ties its embedding and unembedding, as GPT-2 does with HuggingFace's
    weights, until ``enable_compatibility_mode`` folds LayerNorm in and centers the unembedding.
    """

    EMBEDDING = "embedding"
    UNEMBEDDING = "unembedding"


_PROJECTION_WEIGHT = {ProjectionSpace.EMBEDDING: ModelWeight.EMBED, ProjectionSpace.UNEMBEDDING: ModelWeight.UNEMBED}


@dataclass(kw_only=True)
class ParameterLensConfig(LensMethodConfig):
    """``projection`` picks the vocabulary space a weight is read in, and ``scale`` multiplies the weight
    first: a negative scale reads it against its own direction, and the magnitude acts as a temperature."""

    projection: ProjectionSpace = ProjectionSpace.UNEMBEDDING
    scale: float = 1.0


class ParameterLens(AbstractLensMethod[ParameterLensConfig]):
    """Reads a weight vector onto the vocabulary: what a neuron or a head dimension reads or writes.

    A unit is read through the weight ``ACTIVATION_WEIGHT`` names for its activation: the one computing it off
    the residual stream, or, for one no weight computes (``mlp.post``, ``attn.z``), the one writing it back. It
    reads the model's weights rather than a prompt's activations, so a run takes no prompt.
    """

    reads_activations = False

    def extract(
        self,
        lens_input: "LensInput",
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, " d_model"]:
        activation = index.activation()
        if activation not in ACTIVATION_WEIGHT:
            raise ValueError(
                f"{type(self).__name__} cannot project {activation.value}: only MLP-hidden and attention head "
                "locations name a weight vector, one per neuron or head dimension. Residual-stream locations "
                "('embed', 'resid.*', 'attn.out', 'mlp.out') index the residual basis instead of a weight"
            )

        weight = activation_weight(self.model, activation, projection_layer(self.model, index))
        for dim, location in enumerate(ActivationSlicer.get(activation).AXES[2:]):
            weight = weight[(slice(None),) * dim + (_selection(index, location),)]
        return weight.flatten(end_dim=-2).sum(0)

    @torch.no_grad()
    def project(
        self,
        weight: Float[torch.Tensor, " d_model"],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, " d_vocab"]:
        vocabulary = model_weight(self.model, _PROJECTION_WEIGHT[self.method_config.projection])
        return (self.method_config.scale * weight) @ vocabulary.T
