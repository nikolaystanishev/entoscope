import abc

import torch
from jaxtyping import Float

from intex.lens.config import LensConfig
from intex.lens.input import LensInput
from intex.model import Model
from intex.model.config import ModelLocatorConfig
from intex.model.index import IndexElement


class AbstractLensMethod(abc.ABC):
    def __init__(self, model: Model, config: LensConfig):
        self.model = model
        self.config = config

    @abc.abstractmethod
    def extract(
        self,
        lens_input: LensInput,
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "..."]:
        """The tensor one element names."""
        raise NotImplementedError

    @abc.abstractmethod
    def project(
        self,
        source: Float[torch.Tensor, "..."],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "... d_vocab"]:
        """Map one element's tensor to vocabulary logits."""
        raise NotImplementedError
