from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import torch
from jaxtyping import Float

from entoscope.lens.config import LensMethodConfig
from entoscope.lens.method.lens import AbstractLensMethod
from entoscope.lens.method.logit_lens import projection_layer
from entoscope.model.config import ModelLocatorConfig
from entoscope.model.index import IndexElement
from entoscope.model.weight import ACTIVATION_WEIGHT, ModelWeight, activation_weight, model_weight

if TYPE_CHECKING:
    from entoscope.lens.input import LensInput


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
    """The method config of ``parameter_lens``.

    Attributes:
        projection: The vocabulary space a weight is read in.
        scale: What the weight is multiplied by first. A negative scale reads it against its own direction,
            and the magnitude acts as a temperature.
    """

    projection: ProjectionSpace = ProjectionSpace.UNEMBEDDING
    scale: float = 1.0


class ParameterLens(AbstractLensMethod[ParameterLensConfig]):
    """Reads a weight vector onto the vocabulary: what a neuron or a head dimension reads or writes.

    A unit is read through the weight ``ACTIVATION_WEIGHT`` names for its activation: the one computing it
    off the residual stream, or, for one no weight computes (``mlp.post``, ``attn.z``), the one writing it
    back. It reads the model's weights rather than a prompt's activations, so a run takes no prompt.
    """

    reads_activations = False

    def extract(
        self,
        lens_input: "LensInput",
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, " d_model"]:
        """Read the weight vectors of the units an element names, summed into one direction.

        Args:
            lens_input: The input, whose prompt is unused.
            index: The element.

        Returns:
            The summed weight vector.

        Raises:
            ValueError: If the activation names no weight vector, as a residual-stream one does not, or the
                element names no single layer.
        """
        activation = index.activation()
        if activation not in ACTIVATION_WEIGHT:
            raise ValueError(
                f"{type(self).__name__} cannot project {activation.value}: only MLP-hidden and attention head "
                "locations name a weight vector, one per neuron or head dimension. Locations in the residual "
                "stream's basis ('embed', 'resid.*', and a sublayer's input and output: 'attn.in', "
                "'attn.out_pre_norm', 'attn.out', 'mlp.in', 'mlp.out_pre_norm', 'mlp.out') index it instead of a weight"
            )

        weight = activation_weight(self.model, activation, projection_layer(self.model, index))
        return index.read_unit_slices(weight).reshape(-1, weight.shape[-1]).sum(0)

    @torch.no_grad()
    def project(
        self,
        weight: Float[torch.Tensor, " d_model"],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, " d_vocab"]:
        """Read a weight vector in the configured vocabulary space, scaled first.

        Args:
            weight: The element's summed weight vector.
            index: The element.

        Returns:
            The element's logits over the vocabulary.
        """
        vocabulary = model_weight(self.model, _PROJECTION_WEIGHT[self.method_config.projection])
        return (self.method_config.scale * weight) @ vocabulary.T
