"""Probing, run end to end: one probe per location, scored by the metrics the config names."""

import json
import pickle
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from entoscope.model.cache import CachedModel
from entoscope.model.config import Filter, ModelLayerLocation, ModelLocation
from entoscope.probe.config import PoolingType, ProbeConfig, ProbeMetric
from entoscope.probe.input import ProbeInput
from entoscope.probe.prober import Prober

from ..conftest import forbid_model_runs
from .conftest import NEGATIVE, POSITIVE, assert_trained_alike, probe_config, probe_of, refit, run_probe

NEURON = 373
METRICS = [ProbeMetric.ACCURACY, ProbeMetric.F1, ProbeMetric.PRECISION, ProbeMetric.RECALL, ProbeMetric.AUROC]
METRIC_COLUMNS = ["accuracy", "f1", "precision", "recall", "auroc"]

PROBE_YAML = {
    "method": "linear",
    "pool": {"position": "mean"},
    "metric": METRIC_COLUMNS,
    "source": {"activation": "resid.post", "location": ["layer"]},
}


def _single_neuron_config(**overrides) -> ProbeConfig:
    """One feature per example, so the probe cannot separate any four points it likes."""
    return probe_config(
        activation=ModelLayerLocation.MLP_POST,
        location=[ModelLocation.LAYER, ModelLocation.NEURON],
        filter=Filter.from_config([{"neuron": NEURON}]),
        metric=list(METRICS),
        **overrides,
    )


def _neuron_features(caches, layer: int) -> torch.Tensor:
    return torch.stack([cache[f"blocks.{layer}.mlp.hook_post"][0, :, NEURON].mean().reshape(1) for cache in caches])


# --------------------------------------------------------------------------------------------------
# One probe per location
# --------------------------------------------------------------------------------------------------
def test_a_probe_is_trained_for_every_location(model):
    collector = run_probe(model, probe_config())
    result = collector.result

    assert len(result) == model.cfg.n_layers
    assert [column for column in result.columns if not column.startswith("source_")] == ["accuracy"]
    assert set(collector.probes) == {single.output for single in collector.operation_input.index}
    assert len({id(probe) for probe in collector.probes.values()}) == model.cfg.n_layers
    for probe in collector.probes.values():
        assert isinstance(probe, LogisticRegression)
        assert probe.coef_.shape == (1, model.cfg.d_model)


def test_every_probe_is_trained_on_its_own_locations_features(model, caches):
    collector = run_probe(model, _single_neuron_config())

    for layer in range(model.cfg.n_layers):
        assert_trained_alike(probe_of(collector, layer), refit(_neuron_features(caches, layer)))


def test_the_metrics_score_the_probes_own_predictions(model, caches):
    collector = run_probe(model, _single_neuron_config())
    result = collector.result
    labels = np.array([1] * len(POSITIVE) + [0] * len(NEGATIVE))

    assert [column for column in result.columns if not column.startswith("source_")] == METRIC_COLUMNS

    for _, row in result.iterrows():
        features = _neuron_features(caches, row["source_layer"]).numpy()
        probe = probe_of(collector, row["source_layer"])
        predictions, probabilities = probe.predict(features), probe.predict_proba(features)[:, 1]

        assert row["accuracy"] == pytest.approx(accuracy_score(labels, predictions))
        assert row["f1"] == pytest.approx(f1_score(labels, predictions, zero_division=0))
        assert row["precision"] == pytest.approx(precision_score(labels, predictions, zero_division=0))
        assert row["recall"] == pytest.approx(recall_score(labels, predictions, zero_division=0))
        assert row["auroc"] == pytest.approx(roc_auc_score(labels, probabilities))

    # A single neuron is no sentiment detector: the scores have to be imperfect, and auroc — scored on
    # the probabilities rather than the predictions — has to part company with accuracy somewhere.
    assert result["accuracy"].min() < 1.0
    assert (result["auroc"] != result["accuracy"]).any()


def test_a_probe_that_cannot_separate_the_classes_scores_zero(model):
    # The same prompts on both sides: every example has identical features, so the probe predicts a
    # single class and precision, recall and f1 collapse to zero instead of dividing by it.
    collector = run_probe(model, _single_neuron_config(), positive=POSITIVE, negative=POSITIVE)
    row = collector.result.iloc[0]

    assert row["accuracy"] == pytest.approx(0.5)
    assert row["auroc"] == pytest.approx(0.5)
    assert [row["precision"], row["recall"], row["f1"]] == [0.0, 0.0, 0.0]


def test_a_probe_separates_sentiment_at_the_last_layer(model):
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
    config = probe_config(metric=list(METRICS), filter=Filter.from_config([{"layer": -1}]))

    collector = run_probe(model, config, positive=positive, negative=negative)
    result = collector.result

    assert len(result) == 1
    for column in METRIC_COLUMNS:
        assert result.iloc[0][column] >= 0.99

    (probe,) = collector.probes.values()
    assert probe.coef_.shape == (1, model.cfg.d_model)


# --------------------------------------------------------------------------------------------------
# The training fields reach the probes the run returns
# --------------------------------------------------------------------------------------------------
def test_the_training_fields_reach_every_probe(model):
    collector = run_probe(model, _single_neuron_config(max_iter=250, seed=3))

    for probe in collector.probes.values():
        assert probe.max_iter == 250
        assert probe.random_state == 3


def test_the_default_training_fields_apply_when_the_config_omits_them(model, config_from_yaml):
    config = config_from_yaml(ProbeConfig, "my_probe", PROBE_YAML)

    collector = run_probe(model, config)

    for probe in collector.probes.values():
        assert (probe.max_iter, probe.random_state) == (1000, 42)


def test_a_run_through_a_cached_model_reads_the_same_activations(model, tmp_path):
    # Probing runs the model once per example, so each prompt gets its own cache entry.
    cached_model = CachedModel(model, str(tmp_path))
    config = _single_neuron_config()

    first = run_probe(cached_model, config).result

    assert len(list(tmp_path.rglob("*.pt"))) == len(POSITIVE) + len(NEGATIVE)
    pd.testing.assert_frame_equal(first, run_probe(model, config).result)
    pd.testing.assert_frame_equal(run_probe(cached_model, config).result, first)  # now served from disk


# --------------------------------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------------------------------
def test_artifacts_are_saved_to_disk(model, config_from_yaml, tmp_path):
    config = config_from_yaml(ProbeConfig, "my_probe", PROBE_YAML)

    save_dir = tmp_path / "probe_run"
    collector = run_probe(model, config, save_dir=save_dir)

    result_df = pd.read_csv(save_dir / "result.csv")
    assert len(result_df) == len(collector.result)
    for column in METRIC_COLUMNS:
        assert result_df[column].tolist() == pytest.approx(collector.result[column].tolist())

    index_df = pd.read_csv(save_dir / "index.csv")
    assert len(index_df) == len(collector.operation_input.index)

    with open(save_dir / "config.json", encoding="utf-8") as f:
        saved_config = json.load(f)
    # Named by its type alone in the YAML, the probe is written down with its defaults.
    assert saved_config["method"] == {"type": "linear", "max_iter": 1000, "seed": 42}
    assert saved_config["pool"] == {"position": "mean"}
    assert saved_config["metric"] == METRIC_COLUMNS
    assert saved_config["config_id"] == "my_probe"

    with open(save_dir / "probes.pkl", "rb") as f:
        probes = pickle.load(f)
    assert set(probes) == set(collector.probes)
    for output, probe in probes.items():
        assert probe.coef_ == pytest.approx(collector.probes[output].coef_)


# --------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------
def test_a_yaml_config_runs_the_same_way_a_code_built_one_does(model, config_from_yaml):
    from_file = config_from_yaml(ProbeConfig, "my_probe", PROBE_YAML)

    assert from_file.pool == {ModelLocation.POSITION: PoolingType.MEAN}
    assert from_file.metric == list(METRICS)
    assert run_probe(model, from_file).result.equals(run_probe(model, probe_config(metric=list(METRICS))).result)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({**PROBE_YAML, "pool": "mean"}, "Expected a mapping"),
        ({**PROBE_YAML, "pool": None}, "Expected a mapping"),
        ({**PROBE_YAML, "pool": {"layer": "mean"}}, "cannot be pooled"),
        ({**PROBE_YAML, "pool": {"neuron": "mean"}}, "cannot be pooled"),
        ({**PROBE_YAML, "pool": {"position": "median"}}, "median"),
        ({**PROBE_YAML, "pool": {"attention": "mean"}}, "attention"),
        ({**PROBE_YAML, "max_iter": 250}, "Unknown keys for ProbeConfig.*max_iter"),
        ({**PROBE_YAML, "metric": ["accuracy", "mse"]}, "mse"),
        ({**PROBE_YAML, "metric": "accuracy"}, "Expected a list"),
        ({"pool": {"position": "mean"}}, "Missing required keys for ProbeConfig.*method.*metric.*source"),
        ({**PROBE_YAML, "target": {}}, "Unknown keys for ProbeConfig.*target"),
        ({**PROBE_YAML, "method": "mlp"}, "Unknown probe method: mlp"),
        (
            {**PROBE_YAML, "method": {"type": "linear", "max_iter": 250, "solver": "lbfgs"}},
            r"Unknown keys for LinearProbeConfig: \['solver'\]",
        ),
    ],
)
def test_a_malformed_probe_config_is_rejected_when_it_is_loaded(config_from_yaml, raw, message):
    with pytest.raises(ValueError, match=message):
        config_from_yaml(ProbeConfig, "broken", raw)


def test_pooling_defaults_to_nothing_pooled(config_from_yaml):
    raw = {key: value for key, value in PROBE_YAML.items() if key != "pool"}

    assert config_from_yaml(ProbeConfig, "my_probe", raw).pool == {}


def test_a_probe_metric_that_is_not_known_raises(model):
    # A config built in code can hold a metric the dispatch has no implementation for.
    with pytest.raises(ValueError, match="Unknown probe metric"):
        Prober(model).apply(ProbeInput(replace(probe_config(), metric=["mse"]), POSITIVE, NEGATIVE), verbose=False)


def test_both_classes_are_required(model):
    for positive, negative in [(POSITIVE, []), ([], NEGATIVE)]:
        with pytest.raises(ValueError, match="positive and negative prompts"):
            Prober(model).apply(ProbeInput(probe_config(), positive, negative), verbose=False)


# --------------------------------------------------------------------------------------------------
# The plan a run would execute
# --------------------------------------------------------------------------------------------------
def test_a_plan_counts_one_probe_per_location_and_one_pass_per_example(model, monkeypatch, tmp_path):
    save_dir = tmp_path / "planned"
    probe_input = ProbeInput(probe_config(), POSITIVE, NEGATIVE)

    forbid_model_runs(monkeypatch, model)
    plan = Prober(model).plan(probe_input, save_dir=str(save_dir), verbose=False)

    assert not save_dir.exists()
    assert plan.steps == model.cfg.n_layers
    # Features are collected a prompt at a time, so every example is its own cached pass.
    assert plan.forward_passes == len(POSITIVE) + len(NEGATIVE)
    assert [(label, prompt) for label, prompt, _ in plan.prompts] == [
        *(("positive", prompt) for prompt in POSITIVE),
        *(("negative", prompt) for prompt in NEGATIVE),
    ]
    assert plan.activations == sorted(f"blocks.{layer}.hook_resid_post" for layer in range(model.cfg.n_layers))


def test_a_plan_describes_the_run_that_follows_it(model, tmp_path):
    save_dir = tmp_path / "probe_run"
    probe_input = ProbeInput(probe_config(), POSITIVE, NEGATIVE)

    plan = Prober(model).plan(probe_input, save_dir=str(save_dir), verbose=False)
    collector = Prober(model).apply(probe_input, save_dir=str(save_dir), return_collector=True, verbose=False)
    assert collector is not None

    assert len(collector.result) == plan.steps
    assert len(collector.probes) == plan.steps
    assert sorted(path.name for path in save_dir.iterdir()) == sorted(plan.outputs)
    pd.testing.assert_frame_equal(plan.index[0], collector.operation_input.index.to_df())
