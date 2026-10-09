import abc
from typing import Any, ClassVar

import numpy as np
import torch
from jaxtyping import Float, Int

from intex.model import Model
from intex.probe.config import ProbeConfig, ProbeSettings


class AbstractProbeMethod(abc.ABC):
    """A classifier trained on the pooled activations of one location.

    Built afresh for every index row, since each location trains its own probe, and scored on the
    examples it was fitted on. ``settings_class`` names the dataclass of its settings, extending
    ``ProbeSettings``, which the config reads its ``settings`` into as it is built; a probe naming none
    takes ``ProbeSettings`` itself, which no setting fits. ``probe`` is what the collector pickles into
    ``probes.pkl``, so it must pickle.
    """

    settings_class: ClassVar[type[ProbeSettings]] = ProbeSettings

    def __init__(self, model: Model, config: ProbeConfig):
        self.model = model
        self.config = config
        self.settings = config.settings

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
