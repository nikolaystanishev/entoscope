import json
import pickle

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from sklearn.linear_model import LogisticRegression

from intex.model.config import Filter, ModelLayerLocation, ModelLocation, ModelLocatorConfig
from intex.operation.config import ConfigFactory
from intex.probe.config import PoolingType, ProbeConfig, ProbeMetric, ProbeType
from intex.probe.input import ProbeInput
from intex.probe.method.factory import ProbeMethodFactory
from intex.probe.method.probe import AbstractProbeMethod
from intex.probe.prober import Prober

POSITIVE = ["The movie was absolutely wonderful", "I loved every minute of it"]
NEGATIVE = ["The movie was dreadful and boring", "I hated it"]

METRICS = [ProbeMetric.ACCURACY, ProbeMetric.F1, ProbeMetric.PRECISION, ProbeMetric.RECALL, ProbeMetric.AUROC]
METRIC_COLUMNS = ["accuracy", "f1", "precision", "recall", "auroc"]

PROBE_YAML = {
    "my_probe": {
        "type": "linear",
        "pool": {"position": "mean"},
        "metric": METRIC_COLUMNS,
        "source": {"activation": "resid.post", "location": ["layer"]},
    }
}


def _config(source: ModelLocatorConfig | None = None) -> ProbeConfig:
    return ProbeConfig(
        type=ProbeType.LINEAR,
        pool={ModelLocation.POSITION: PoolingType.MEAN},
        metric=list(METRICS),
        source=source
        or ModelLocatorConfig(activation=ModelLayerLocation.RESIDUAL_POST, location=[ModelLocation.LAYER]),
    )


class FakeProbeMethod(AbstractProbeMethod):
    """Predicts all-positive with constant 0.5 scores so every metric has a closed form."""

    instances: list["FakeProbeMethod"] = []

    def __init__(self, model, config):
        super().__init__(model, config)
        self.fitted: tuple[torch.Tensor, torch.Tensor] | None = None
        self._probe = object()
        type(self).instances.append(self)

    def fit(self, features, labels):
        self.fitted = (features, labels)

    def predict(self, features):
        return np.ones(len(features), dtype=int)

    def predict_proba(self, features):
        return np.full(len(features), 0.5)

    @property
    def probe(self):
        return self._probe


# With 2 positive and 2 negative examples, all-positive predictions with constant scores give:
EXPECTED_FAKE_METRICS = {"accuracy": 0.5, "f1": 2 / 3, "precision": 0.5, "recall": 1.0, "auroc": 0.5}


@pytest.fixture
def fake_method(monkeypatch) -> type[FakeProbeMethod]:
    monkeypatch.setattr(ProbeMethodFactory, "get", staticmethod(lambda probe_type: FakeProbeMethod))
    FakeProbeMethod.instances = []
    return FakeProbeMethod


def test_metrics_computed_per_location(model, fake_method):
    probe_input = ProbeInput(POSITIVE, NEGATIVE, _config())
    collector = Prober(model).apply(probe_input, return_collector=True, verbose=False)
    result = collector.result

    assert len(result) == model.cfg.n_layers
    metric_columns = [col for col in result.columns if not col.startswith(("source_", "index_"))]
    assert metric_columns == METRIC_COLUMNS

    for _, row in result.iterrows():
        for column in METRIC_COLUMNS:
            assert row[column] == pytest.approx(EXPECTED_FAKE_METRICS[column])


def test_fresh_probe_per_location_trained_on_pooled_features(model, fake_method):
    probe_input = ProbeInput(POSITIVE, NEGATIVE, _config())
    collector = Prober(model).apply(probe_input, return_collector=True, verbose=False)

    assert len(fake_method.instances) == len(probe_input.index)
    assert len(set(map(id, fake_method.instances))) == len(fake_method.instances)
    assert collector.probes == {
        single.output: instance._probe
        for single, instance in zip(probe_input.index, fake_method.instances, strict=True)
    }
    for instance in fake_method.instances:
        features, labels = instance.fitted
        assert features.shape == (4, model.cfg.d_model)
        assert labels.tolist() == [1, 1, 0, 0]


def test_artifacts_saved_to_disk(model, fake_method, tmp_path):
    config_yaml = tmp_path / "configs.yaml"
    config_yaml.write_text(yaml.safe_dump(PROBE_YAML))
    config = ConfigFactory[ProbeConfig](str(config_yaml)).create("my_probe")

    save_dir = tmp_path / "probe_run"
    probe_input = ProbeInput(POSITIVE, NEGATIVE, config)
    collector = Prober(model).apply(probe_input, save_dir=str(save_dir), return_collector=True, verbose=False)

    result_df = pd.read_csv(save_dir / "result.csv")
    assert len(result_df) == len(collector.result)
    for column in METRIC_COLUMNS:
        assert result_df[column].tolist() == pytest.approx(collector.result[column].tolist())

    index_df = pd.read_csv(save_dir / "index.csv")
    assert len(index_df) == len(probe_input.index)

    with open(save_dir / "config.json", encoding="utf-8") as f:
        saved_config = json.load(f)
    assert saved_config["type"] == "linear"
    assert saved_config["pool"] == {"position": "mean"}
    assert saved_config["metric"] == METRIC_COLUMNS
    assert saved_config["config_id"] == "my_probe"

    with open(save_dir / "probes.pkl", "rb") as f:
        probes = pickle.load(f)
    assert set(probes) == set(collector.probes)


def test_linear_probe_separates_sentiment(model):
    positive = [
        "The movie was absolutely wonderful",
        "I loved every minute of it",
        "What a fantastic experience",
        "The acting was brilliant",
    ]
    negative = [
        "The movie was absolutely terrible",
        "I hated every minute of it",
        "What a dreadful experience",
        "The acting was awful",
    ]
    config = _config(
        source=ModelLocatorConfig(
            activation=ModelLayerLocation.RESIDUAL_POST,
            location=[ModelLocation.LAYER],
            filter=Filter(inclusion_terms=[{ModelLocation.LAYER: -1}]),
        )
    )
    collector = Prober(model).apply(ProbeInput(positive, negative, config), return_collector=True, verbose=False)
    result = collector.result

    assert len(result) == 1
    for column in METRIC_COLUMNS:
        assert result.iloc[0][column] >= 0.99

    (probe,) = collector.probes.values()
    assert isinstance(probe, LogisticRegression)
    assert probe.coef_.shape == (1, model.cfg.d_model)
