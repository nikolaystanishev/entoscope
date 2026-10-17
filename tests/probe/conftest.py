import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from entoscope.model.config import ModelLayerLocation, ModelLocation, ModelLocatorConfig
from entoscope.probe.collector import ProbeCollector
from entoscope.probe.config import PoolingType, ProbeConfig, ProbeMetric
from entoscope.probe.input import ProbeInput
from entoscope.probe.method.linear import LinearProbeConfig
from entoscope.probe.prober import Prober

POSITIVE = ["The movie was absolutely wonderful", "I loved every minute of it"]
NEGATIVE = ["The movie was dreadful and boring", "I hated every minute of it"]

# Two tokens shorter, so any grid over positions has to stay symbolic for these.
SHORT_POSITIVE = ["I loved it", "It was great"]
SHORT_NEGATIVE = ["I hated it", "It was awful"]

MEAN_OVER_POSITION = {ModelLocation.POSITION: PoolingType.MEAN}


def probe_config(
    activation: ModelLayerLocation = ModelLayerLocation.RESIDUAL_POST,
    location: list[ModelLocation] | None = None,
    pool: dict | None = None,
    metric: list[ProbeMetric] | None = None,
    max_iter: int = 1000,
    seed: int = 42,
    **source,
) -> ProbeConfig:
    return ProbeConfig(
        method=LinearProbeConfig(type="linear", max_iter=max_iter, seed=seed),
        pool=MEAN_OVER_POSITION if pool is None else pool,
        metric=metric or [ProbeMetric.ACCURACY],
        source=ModelLocatorConfig(activation=activation, location=location or [ModelLocation.LAYER], **source),
    )


def run_probe(model, config: ProbeConfig, positive=None, negative=None, save_dir=None) -> ProbeCollector:
    probe_input = ProbeInput(config, positive or POSITIVE, negative or NEGATIVE)
    collector = Prober(model).apply(
        probe_input, save_dir=None if save_dir is None else str(save_dir), return_collector=True, verbose=False
    )
    assert collector is not None
    return collector


def probe_of(collector: ProbeCollector, layer: int = 0, **coordinates) -> LogisticRegression:
    """The trained probe of one location, taken from what the run returns."""
    output = next(
        single.output
        for single in collector.operation_input.index
        if single.source.layer() == layer
        and all(single.source.get(ModelLocation(name)) == value for name, value in coordinates.items())
    )
    return collector.probes[output]


def refit(features: torch.Tensor | np.ndarray, labels=None, max_iter: int = 1000, seed: int = 42) -> LogisticRegression:
    """A probe trained the way the run trains one, on features computed by hand."""
    if labels is None:
        labels = np.array([1] * len(POSITIVE) + [0] * len(NEGATIVE))
    values = features.detach().cpu().numpy() if isinstance(features, torch.Tensor) else np.asarray(features)
    return LogisticRegression(max_iter=max_iter, random_state=seed).fit(values, labels)


def assert_trained_alike(trained: LogisticRegression, expected: LogisticRegression) -> None:
    assert trained.coef_.shape == expected.coef_.shape
    assert trained.coef_ == pytest.approx(expected.coef_, rel=1e-4, abs=1e-6)
    assert trained.intercept_ == pytest.approx(expected.intercept_, rel=1e-4, abs=1e-6)


def activation_caches(model, prompts: list[str] | None = None) -> list:
    """Every prompt's activations from a plain forward pass, in the order the labels are built."""
    caches = []
    for prompt in POSITIVE + NEGATIVE if prompts is None else prompts:
        with torch.no_grad():
            _, cache = model.run_with_cache(model.to_tokens(prompt))
        caches.append(cache)
    return caches


@pytest.fixture(scope="module")
def caches(model):
    return activation_caches(model)


@pytest.fixture(scope="module")
def small_caches(small_model):
    return activation_caches(small_model)
