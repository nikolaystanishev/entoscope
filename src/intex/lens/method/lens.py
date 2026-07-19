import abc

import torch
from jaxtyping import Float
from transformer_lens import HookedTransformer

from intex.lens.config import LensConfig
from intex.model.config import ModelLocatorConfig
from intex.model.index import IndexElement


class AbstractLensMethod(abc.ABC):
    def __init__(self, model: HookedTransformer, config: LensConfig):
        self.model = model
        self.config = config

    @abc.abstractmethod
    def project(
        self,
        activation: Float[torch.Tensor, "..."],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "... d_vocab"]:
        """Map one element's activation slice to vocabulary logits."""
        raise NotImplementedError
