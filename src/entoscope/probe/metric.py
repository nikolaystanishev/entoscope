from __future__ import annotations

import abc

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from entoscope.probe.config import ProbeMetric


class AbstractProbeMetric(abc.ABC):
    """A score of a probe's predictions against the labels."""

    @classmethod
    @abc.abstractmethod
    def compute(cls, labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> float:
        """Score a probe's predictions.

        Args:
            labels: The 0/1 labels.
            predictions: The probe's 0/1 predictions.
            probabilities: The probe's positive-class probabilities.

        Returns:
            The score.
        """
        raise NotImplementedError

    @staticmethod
    def get(metric: ProbeMetric) -> type[AbstractProbeMetric]:
        """Pick the implementation of a metric.

        Args:
            metric: The metric.

        Returns:
            The metric's class.

        Raises:
            ValueError: If the metric is unknown.
        """
        match metric:
            case ProbeMetric.ACCURACY:
                return AccuracyMetric
            case ProbeMetric.F1:
                return F1Metric
            case ProbeMetric.PRECISION:
                return PrecisionMetric
            case ProbeMetric.RECALL:
                return RecallMetric
            case ProbeMetric.AUROC:
                return AurocMetric
            case _:
                raise ValueError(f"Unknown probe metric: {metric}")


class AccuracyMetric(AbstractProbeMetric):
    """The share of predictions that match their label."""

    @classmethod
    def compute(cls, labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> float:
        return float(accuracy_score(labels, predictions))


class F1Metric(AbstractProbeMetric):
    """The harmonic mean of precision and recall, 0 when there are no positive predictions."""

    @classmethod
    def compute(cls, labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> float:
        return float(f1_score(labels, predictions, zero_division=0))


class PrecisionMetric(AbstractProbeMetric):
    """The share of positive predictions that are positive, 0 when there are none."""

    @classmethod
    def compute(cls, labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> float:
        return float(precision_score(labels, predictions, zero_division=0))


class RecallMetric(AbstractProbeMetric):
    """The share of positive examples predicted positive."""

    @classmethod
    def compute(cls, labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> float:
        return float(recall_score(labels, predictions, zero_division=0))


class AurocMetric(AbstractProbeMetric):
    """The area under the ROC curve of the positive-class probabilities."""

    @classmethod
    def compute(cls, labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> float:
        return float(roc_auc_score(labels, probabilities))
