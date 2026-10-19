import abc
from typing import TYPE_CHECKING, ClassVar, Generic, cast

import torch
from jaxtyping import Float
from typing_extensions import TypeVar

from entoscope.lens.config import LensConfig, LensMethodConfig
from entoscope.model import Model
from entoscope.model.config import ModelLocatorConfig
from entoscope.model.index import IndexElement

if TYPE_CHECKING:
    from entoscope.lens.input import LensInput

LensMethodConfigT = TypeVar("LensMethodConfigT", bound=LensMethodConfig, default=LensMethodConfig)


class AbstractLensMethod(abc.ABC, Generic[LensMethodConfigT]):
    """The base of every lens: reads each index element out onto the vocabulary.

    A lens is built once per run, from the run's config alone, so a lens that handles several kinds of
    activation tells them apart through ``index.activation()`` itself. Its type argument names the
    dataclass of its method config, extending ``LensMethodConfig``, which a config file's ``method`` is
    read into. A lens naming none takes ``LensMethodConfig`` itself, which no field fits.

    Args:
        model: The model the lens reads.
        config: The run's config.

    Attributes:
        reads_activations: ``False`` for a lens that reads only the model's weights, which takes no
            prompt and has no ``position`` axis.
        method_config: The run's method config, typed by the lens's type argument.
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
        """Read the tensor one element names.

        Args:
            lens_input: The prepared input, whose cache holds the activations the index reads.
            index: The element.

        Returns:
            The element's activation, or the weight it names.

        Raises:
            ValueError: If the lens cannot read the element.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def project(
        self,
        source: Float[torch.Tensor, "..."],
        index: IndexElement[ModelLocatorConfig],
    ) -> Float[torch.Tensor, "... d_vocab"]:
        """Map one element's tensor onto the vocabulary.

        The collector ranks the tokens of a single slice, so an element projects to a single ``d_vocab``
        vector.

        Args:
            source: What ``extract`` returned for the element.
            index: The element.

        Returns:
            The element's logits over the vocabulary.

        Raises:
            ValueError: If the lens cannot read the element.
        """
        raise NotImplementedError
