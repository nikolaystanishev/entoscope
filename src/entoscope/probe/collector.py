import os
import pickle
from typing import Any

from entoscope.model import Model
from entoscope.model.index import Index
from entoscope.operation.collector import FrameCollector
from entoscope.probe.config import ProbeMetric
from entoscope.probe.input import ProbeInput
from entoscope.probe.method.probe import AbstractProbeMethod


class ProbeCollector(FrameCollector[ProbeInput]):
    """One row per location, with a column per metric the config names.

    Writes ``result.csv``, ``index.csv``, ``config.json`` and ``probes.pkl``: the fitted probes, pickled
    as a dict from result row to estimator.
    """

    def __init__(self, operation_input: ProbeInput, model: Model, save_dir: str | None = None):
        super().__init__(operation_input, model, save_dir)
        self._probes: dict[int, Any] = {}

    @property
    def probes(self) -> dict[int, Any]:
        return self._probes

    def collect(self, method: AbstractProbeMethod, metric_values: dict[ProbeMetric, float], index: int):
        self._collect_cell(index, {metric.value: value for metric, value in metric_values.items()})
        self._probes[index] = method.probe

    def _indices(self) -> dict[str, Index]:
        return {"index.csv": self.operation_input.index}

    def _save_result(self, save_dir: str):
        super()._save_result(save_dir)
        with open(os.path.join(save_dir, "probes.pkl"), "wb") as f:
            pickle.dump(self.probes, f)

    def output_files(self) -> list[str]:
        return ["probes.pkl", *super().output_files()]
