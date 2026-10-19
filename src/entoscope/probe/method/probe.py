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
    """The base of every probe: a classifier trained on the pooled activations of one location.

    Built afresh for every index row, since each location trains its own probe, and scored on the
    examples it was fitted on. Its type argument names the dataclass of its method config, extending
    ``ProbeMethodConfig``, which a config file's ``method`` is read into. A probe naming none takes
    ``ProbeMethodConfig`` itself, which no field fits.

    Args:
        model: The model the features are read from.
        config: The run's config.

    Attributes:
        method_config: The run's method config, typed by the probe's type argument.
    """

    def __init__(self, model: Model, config: ProbeConfig):
        self.model = model
        self.config = config
        self.method_config = cast(ProbeMethodConfigT, config.method)

    @abc.abstractmethod
    def fit(self, features: Float[torch.Tensor, "examples features"], labels: Int[torch.Tensor, " examples"]):
        """Train the probe.

        Args:
            features: One feature vector per example.
            labels: Every example's 0/1 label.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, features: Float[torch.Tensor, "examples features"]) -> np.ndarray:
        """Predict labels.

        Args:
            features: One feature vector per example.

        Returns:
            Every example's predicted 0/1 label.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict_proba(self, features: Float[torch.Tensor, "examples features"]) -> np.ndarray:
        """Predict the positive class's probability.

        Args:
            features: One feature vector per example.

        Returns:
            Every example's positive-class probability.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def probe(self) -> Any:
        """The fitted estimator, which the collector pickles into ``probes.pkl``, so it must pickle."""
        raise NotImplementedError
