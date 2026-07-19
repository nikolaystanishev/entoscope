import os
import pickle
from collections.abc import Sequence
from typing import Any

import pandas as pd

from intex.probe.config import ProbeMetric
from intex.probe.input import ProbeInput
from intex.probe.method.probe import AbstractProbeMethod
from intex.util.collections import dataframe_to_list, to_dict
from intex.util.data import create_if_not_exists, write_json


class ProbeCollector:
    def __init__(self, probe_input: ProbeInput, save_dir: str | None = None):
        self.probe_input = probe_input
        self.save_dir = save_dir

        self._cells: dict[tuple[int, ...], dict[str, float]] | None = None
        self._probes: dict[tuple[int, ...], Any] | None = None

    @property
    def result(self) -> pd.DataFrame:
        assert self._cells is not None
        rows = pd.DataFrame([self._cells[tuple(row)] for row in dataframe_to_list(self.probe_input.index.output_df)])
        return pd.concat([self.probe_input.index.to_df().reset_index(drop=True), rows], axis=1)

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

    def save(self):
        if self.save_dir is not None:
            create_if_not_exists(self.save_dir)

            self.result.to_csv(os.path.join(self.save_dir, "result.csv"), index=False)
            self.probe_input.index.to_csv(os.path.join(self.save_dir, "index.csv"))
            write_json(os.path.join(self.save_dir, "config.json"), to_dict(self.probe_input.config))
            with open(os.path.join(self.save_dir, "probes.pkl"), "wb") as f:
                pickle.dump(self.probes, f)

            print(f"{self.probe_input.config.config_id} done - results saved to {self.save_dir}")
