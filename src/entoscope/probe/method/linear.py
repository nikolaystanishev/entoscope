from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float, Int
from sklearn.linear_model import LogisticRegression

from entoscope.model import Model
from entoscope.probe.config import ProbeConfig, ProbeMethodConfig
from entoscope.probe.method.probe import AbstractProbeMethod


@dataclass(kw_only=True)
class LinearProbeConfig(ProbeMethodConfig):
    """A logistic regression: ``max_iter`` bounds its solver and ``seed`` fixes its randomness."""

    max_iter: int = 1000
    seed: int = 42


class LinearProbe(AbstractProbeMethod[LinearProbeConfig]):
    def __init__(self, model: Model, config: ProbeConfig):
        super().__init__(model, config)
        self._probe = LogisticRegression(max_iter=self.method_config.max_iter, random_state=self.method_config.seed)

    def fit(self, features: Float[torch.Tensor, "examples features"], labels: Int[torch.Tensor, " examples"]):
        self._probe.fit(features.detach().cpu().numpy(), labels.detach().cpu().numpy())

    def predict(self, features: Float[torch.Tensor, "examples features"]) -> np.ndarray:
        return self._probe.predict(features.detach().cpu().numpy())

    def predict_proba(self, features: Float[torch.Tensor, "examples features"]) -> np.ndarray:
        return self._probe.predict_proba(features.detach().cpu().numpy())[:, 1]

    @property
    def probe(self) -> LogisticRegression:
        return self._probe
