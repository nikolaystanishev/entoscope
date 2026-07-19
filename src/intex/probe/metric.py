from __future__ import annotations

import abc

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from intex.probe.config import ProbeMetric


class AbstractProbeMetric(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def compute(cls, labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> float:
        """Score 0/1 ``predictions`` (or positive-class ``probabilities``) against 0/1 ``labels``."""
        raise NotImplementedError

    @staticmethod
    def get(metric: ProbeMetric) -> type[AbstractProbeMetric]:
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
    @classmethod
    def compute(cls, labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> float:
        return float(accuracy_score(labels, predictions))


class F1Metric(AbstractProbeMetric):
    @classmethod
    def compute(cls, labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> float:
        return float(f1_score(labels, predictions, zero_division=0))


class PrecisionMetric(AbstractProbeMetric):
    @classmethod
    def compute(cls, labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> float:
        return float(precision_score(labels, predictions, zero_division=0))


class RecallMetric(AbstractProbeMetric):
    @classmethod
    def compute(cls, labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> float:
        return float(recall_score(labels, predictions, zero_division=0))


class AurocMetric(AbstractProbeMetric):
    @classmethod
    def compute(cls, labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> float:
        return float(roc_auc_score(labels, probabilities))
