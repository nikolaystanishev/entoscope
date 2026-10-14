import abc
from typing import Any, Generic, cast

import numpy as np
import torch
from jaxtyping import Float, Int
from typing_extensions import TypeVar

from entoscope.model import Model
from entoscope.probe.config import ProbeConfig, ProbeMethodConfig

ProbeMethodConfigT = TypeVar("ProbeMethodConfigT", bound=ProbeMethodConfig, default=ProbeMethodConfig)


class AbstractProbeMethod(abc.ABC, Generic[ProbeMethodConfigT]):
    """A classifier trained on the pooled activations of one location.

    Built afresh for every index row, since each location trains its own probe, and scored on the
    examples it was fitted on. Its type argument names the dataclass of its method config, extending
    ``ProbeMethodConfig``, which a config file's ``method`` is read into; a probe naming none
    takes ``ProbeMethodConfig`` itself, which no field fits. ``probe`` is what the collector pickles into
    ``probes.pkl``, so it must pickle.
    """

    def __init__(self, model: Model, config: ProbeConfig):
        self.model = model
        self.config = config
        self.method_config = cast(ProbeMethodConfigT, config.method)

    @abc.abstractmethod
    def fit(self, features: Float[torch.Tensor, "examples features"], labels: Int[torch.Tensor, " examples"]):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, features: Float[torch.Tensor, "examples features"]) -> np.ndarray:
        """Predicted 0/1 labels."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict_proba(self, features: Float[torch.Tensor, "examples features"]) -> np.ndarray:
        """Positive-class probabilities."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def probe(self) -> Any:
        """The fitted estimator, for the collector."""
        raise NotImplementedError
