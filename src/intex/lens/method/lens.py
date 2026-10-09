import abc
from typing import TYPE_CHECKING, ClassVar

import torch
from jaxtyping import Float

from intex.lens.config import LensConfig, LensSettings
from intex.model import Model
from intex.model.config import ModelLocatorConfig
from intex.model.index import IndexElement

if TYPE_CHECKING:  # ``intex.lens.input`` looks the lens up through the factory, which imports this module.
    from intex.lens.input import LensInput


class AbstractLensMethod(abc.ABC):
    """Reads each index element out onto the vocabulary.

    A lens is built once per run, from the run's config alone, so a lens that handles several kinds of
    activation tells them apart through ``index.activation()`` itself. ``settings_class`` names the
    dataclass of its settings, extending ``LensSettings``, which the config reads its ``settings`` into as
    it is built; a lens naming none takes ``LensSettings`` itself, which no setting fits. ``extract``
    returns the tensor an element names and ``project`` maps it onto the vocabulary; the collector ranks
    the tokens of a single slice, so an element projects to a single ``d_vocab`` vector.
    ``reads_activations`` is ``False`` for a lens that reads only the model's weights: it takes no prompt
    and has no ``position`` axis.
    """

    reads_activations: ClassVar[bool] = True
    settings_class: ClassVar[type[LensSettings]] = LensSettings

    def __init__(self, model: Model, config: LensConfig):
        self.model = model
        self.config = config
        self.settings = config.settings

    @abc.abstractmethod
    def extract(
        self,
        lens_input: "LensInput",
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
