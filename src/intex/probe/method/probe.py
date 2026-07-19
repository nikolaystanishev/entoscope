import abc
from typing import Any

import numpy as np
import torch
from jaxtyping import Float, Int
from transformer_lens import HookedTransformer

from intex.probe.config import ProbeConfig


class AbstractProbeMethod(abc.ABC):
    def __init__(self, model: HookedTransformer, config: ProbeConfig):
        self.model = model
        self.config = config

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
