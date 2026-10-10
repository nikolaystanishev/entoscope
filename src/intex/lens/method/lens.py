import abc
from typing import TYPE_CHECKING, ClassVar, Generic, cast

import torch
from jaxtyping import Float
from typing_extensions import TypeVar

from intex.lens.config import LensConfig, LensMethodConfig
from intex.model import Model
from intex.model.config import ModelLocatorConfig
from intex.model.index import IndexElement

if TYPE_CHECKING:  # ``intex.lens.input`` looks the lens up through the factory, which imports this module.
    from intex.lens.input import LensInput

LensMethodConfigT = TypeVar("LensMethodConfigT", bound=LensMethodConfig, default=LensMethodConfig)


class AbstractLensMethod(abc.ABC, Generic[LensMethodConfigT]):
    """Reads each index element out onto the vocabulary.

    A lens is built once per run, from the run's config alone, so a lens that handles several kinds of
    activation tells them apart through ``index.activation()`` itself. Its type argument names the
    dataclass of its method config, extending ``LensMethodConfig``, which a config file's ``method`` is read
    into; a lens naming none takes ``LensMethodConfig`` itself, which no field fits. ``extract``
    returns the tensor an element names and ``project`` maps it onto the vocabulary; the collector ranks
    the tokens of a single slice, so an element projects to a single ``d_vocab`` vector. Either raises for
    an element the lens cannot read. ``reads_activations`` is ``False`` for a lens that reads only the
    model's weights: it takes no prompt and has no ``position`` axis.
    """

    reads_activations: ClassVar[bool] = True

    def __init__(self, model: Model, config: LensConfig):
        self.model = model
        self.config = config
        self.method_config = cast(LensMethodConfigT, config.method)

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
