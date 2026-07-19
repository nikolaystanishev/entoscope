import numpy as np
import pytest

from intex.probe.config import ProbeMetric
from intex.probe.metric import (
    AbstractProbeMetric,
    AccuracyMetric,
    AurocMetric,
    F1Metric,
    PrecisionMetric,
    RecallMetric,
)

LABELS = np.array([1, 1, 0, 0])
PREDICTIONS = np.array([1, 0, 0, 0])
PROBABILITIES = np.array([0.9, 0.4, 0.6, 0.1])


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        (ProbeMetric.ACCURACY, AccuracyMetric),
        (ProbeMetric.F1, F1Metric),
        (ProbeMetric.PRECISION, PrecisionMetric),
        (ProbeMetric.RECALL, RecallMetric),
        (ProbeMetric.AUROC, AurocMetric),
    ],
)
def test_get(metric, expected):
    assert AbstractProbeMetric.get(metric) is expected


def test_get_unknown_raises():
    with pytest.raises(ValueError, match="Unknown probe metric"):
        AbstractProbeMetric.get("mse")


@pytest.mark.parametrize(
    ("metric_class", "expected"),
    [
        (AccuracyMetric, 0.75),
        (PrecisionMetric, 1.0),
        (RecallMetric, 0.5),
        (F1Metric, 2 / 3),
        (AurocMetric, 0.75),
    ],
)
def test_hand_computed_values(metric_class, expected):
    assert metric_class.compute(LABELS, PREDICTIONS, PROBABILITIES) == pytest.approx(expected)


def test_single_class_predictions_score_zero():
    predictions = np.zeros(4, dtype=int)
    for metric_class in (PrecisionMetric, RecallMetric, F1Metric):
        assert metric_class.compute(LABELS, predictions, PROBABILITIES) == 0.0
