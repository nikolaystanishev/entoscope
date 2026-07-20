import os
import pickle
from collections.abc import Sequence
from typing import Any

import pandas as pd

from intex.operation.collector import AbstractCollector
from intex.probe.config import ProbeMetric
from intex.probe.input import ProbeInput
from intex.probe.method.probe import AbstractProbeMethod
from intex.util.collections import dataframe_to_list


class ProbeCollector(AbstractCollector[ProbeInput]):
    def __init__(self, probe_input: ProbeInput, save_dir: str | None = None):
        super().__init__(probe_input, save_dir)

        self._cells: dict[tuple[int, ...], dict[str, float]] | None = None
        self._probes: dict[tuple[int, ...], Any] | None = None

    @property
    def result(self) -> pd.DataFrame:
        assert self._cells is not None
        rows = pd.DataFrame(
            [self._cells[tuple(row)] for row in dataframe_to_list(self.operation_input.index.output_df)]
        )
        return pd.concat([self.operation_input.index.to_df().reset_index(drop=True), rows], axis=1)

    @property
    def probes(self) -> dict[tuple[int, ...], Any]:
        assert self._probes is not None
        return self._probes

    def prepare(self):
        self._cells = {}
        self._probes = {}

    def collect(self, method: AbstractProbeMethod, metric_values: dict[ProbeMetric, float], index: Sequence[int]):
        assert self._cells is not None and self._probes is not None
        self._cells[tuple(index)] = {metric.value: value for metric, value in metric_values.items()}
        self._probes[tuple(index)] = method.probe

    def _save_result(self, save_dir: str):
        self.result.to_csv(os.path.join(save_dir, "result.csv"), index=False)
        self.operation_input.index.to_csv(os.path.join(save_dir, "index.csv"))
        with open(os.path.join(save_dir, "probes.pkl"), "wb") as f:
            pickle.dump(self.probes, f)
