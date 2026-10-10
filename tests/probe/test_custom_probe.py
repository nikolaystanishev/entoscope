"""A probe a user registers, trained and scored end to end the way the built-in one is."""

import json
import pickle
from dataclasses import dataclass

import pytest
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from intex.probe.config import ProbeConfig, ProbeMethodConfig
from intex.probe.method.factory import ProbeMethodFactory
from intex.probe.method.probe import AbstractProbeMethod

from .conftest import NEGATIVE, POSITIVE, probe_of, run_probe

TEMPERATURE = 0.5
CENTROID_YAML = {
    "method": {"type": "centroid", "temperature": TEMPERATURE},
    "pool": {"position": "mean"},
    "metric": ["accuracy", "auroc"],
    "source": {"activation": "resid.post", "location": ["layer"]},
}


@dataclass(kw_only=True)
class CentroidProbeConfig(ProbeMethodConfig):
    temperature: float = 1.0


class CentroidProbe(AbstractProbeMethod[CentroidProbeConfig]):
    """Labels an example by the nearer of the two class means."""

    def fit(self, features, labels):
        self._centroids = torch.stack([features[labels == label].mean(0) for label in (0, 1)])

    def predict(self, features):
        return self._distances(features).argmin(-1).numpy()

    def predict_proba(self, features):
        return (-self._distances(features) / self.method_config.temperature).softmax(-1)[:, 1].numpy()

    @property
    def probe(self):
        return {"centroids": self._centroids, "temperature": self.method_config.temperature}

    def _distances(self, features):
        return torch.cdist(features, self._centroids)


@pytest.fixture
def centroid_probe(registries):
    return ProbeMethodFactory.register("centroid")(CentroidProbe)


def test_a_registered_probe_is_trained_and_scored_at_every_location(model, caches, config_from_yaml, centroid_probe):
    config = config_from_yaml(ProbeConfig, "centroids", CENTROID_YAML)
    labels = torch.tensor([1] * len(POSITIVE) + [0] * len(NEGATIVE))

    collector = run_probe(model, config)

    assert len(collector.result) == model.cfg.n_layers
    for _, row in collector.result.iterrows():
        # The run pools each prompt's residual stream over its positions, the BOS token included.
        layer = row["source_layer"]
        features = torch.stack([cache[f"blocks.{layer}.hook_resid_post"][0].mean(0) for cache in caches])
        centroids = torch.stack([features[labels == label].mean(0) for label in (0, 1)])
        distances = torch.cdist(features, centroids)

        trained = probe_of(collector, layer)
        assert torch.allclose(trained["centroids"], centroids)
        assert trained["temperature"] == TEMPERATURE
        assert row["accuracy"] == pytest.approx(accuracy_score(labels, distances.argmin(-1)))
        assert row["auroc"] == pytest.approx(roc_auc_score(labels, (-distances / TEMPERATURE).softmax(-1)[:, 1]))


def test_a_registered_probe_leaves_its_estimators_and_method_config_behind(
    model, config_from_yaml, centroid_probe, tmp_path
):
    save_dir = tmp_path / "centroids"

    collector = run_probe(model, config_from_yaml(ProbeConfig, "centroids", CENTROID_YAML), save_dir=save_dir)

    with open(save_dir / "probes.pkl", "rb") as f:
        probes = pickle.load(f)
    assert set(probes) == set(collector.probes)
    for output, probe in probes.items():
        assert torch.equal(probe["centroids"], collector.probes[output]["centroids"])
    saved = json.loads((save_dir / "config.json").read_text())
    assert saved["method"] == {"type": "centroid", "temperature": TEMPERATURE}
    assert saved["metric"] == ["accuracy", "auroc"]
