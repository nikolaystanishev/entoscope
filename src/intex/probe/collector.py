import os
import pickle
from typing import Any

from intex.model.index import Index
from intex.operation.collector import FrameCollector
from intex.probe.config import ProbeMetric
from intex.probe.input import ProbeInput
from intex.probe.method.probe import AbstractProbeMethod


class ProbeCollector(FrameCollector[ProbeInput]):
    _probes: dict[int, Any] | None = None

    @property
    def probes(self) -> dict[int, Any]:
        assert self._probes is not None
        return self._probes

    def prepare(self):
        super().prepare()
        self._probes = {}

    def collect(self, method: AbstractProbeMethod, metric_values: dict[ProbeMetric, float], index: int):
        assert self._probes is not None
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
