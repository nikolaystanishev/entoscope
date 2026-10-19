"""Interactive Plotly rendering of a patching run.

The grid is the run's own index, layer up and position across for the canonical sweep, coloured by
the metric on a diverging scale centred on the value an ineffective patch would produce, so warm
cells are the locations that moved the model toward the source. One such grid is drawn per column
the config names, each centred on its own comparison's ineffective value. A cell's hover names the
token its patched run predicts next, and the subtitle the token each unpatched prompt does.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape

import pandas as pd

from entoscope.model import Model
from entoscope.model.config import POSITION_AXES
from entoscope.model.index import Index
from entoscope.model.token import str_token, str_token_labels
from entoscope.operation.visualization import DIVERGING, GridFigure, describe_row, diverging_range, format_value, titled
from entoscope.patching.config import MetricComparison, MetricPair, metric_column
from entoscope.patching.input import PatchingInput


class PatchingFigure(GridFigure[PatchingInput, MetricPair]):
    """The page of a patching run: one grid per result column, each panel a metric and its comparison.

    The patch elements the result is not laid out along are described in the subtitle when they patch
    the same location in every run, and in each cell's hover when they vary with the row.

    Args:
        result: The run's result frame.
        operation_input: The input the run iterated.
        model: The model the run patched.
        index: The index that laid the result out.
    """

    colorscale = DIVERGING

    def __init__(self, result: pd.DataFrame, operation_input: PatchingInput, model: Model, index: Index) -> None:
        super().__init__(result, operation_input, model, index)
        self.constant, self.per_row = self._companions()

    @property
    def title(self) -> str:
        return titled("Patching", self.operation_input.config)

    def panels(self) -> list[MetricPair]:
        return self.operation_input.config.metric_pairs()

    def heading(self, panel: MetricPair) -> str:
        config = self.operation_input.config
        return titled(f"Patching - {config.method.type} · {self.value_column(panel)}", config)

    def subtitle(self, panel: MetricPair) -> str:
        patching_input = self.operation_input
        prompts = self._prompt("target", patching_input.target_prompt, patching_input.target_generation)
        if patching_input.source_prompt is not None:
            source = self._prompt("source", patching_input.source_prompt, patching_input.source_generation)
            prompts = f"{source} → {prompts}"
        parts = [prompts]
        if self.constant:
            parts.append(self.constant)
        return " · ".join(parts)

    def value_column(self, panel: MetricPair) -> str:
        return metric_column(*panel)

    def value_range(self, panel: MetricPair, values: Sequence[float]) -> tuple[float, float]:
        metric, comparison = panel
        midpoint = (
            float(self.operation_input.target_baseline(metric)) if comparison is MetricComparison.ABSOLUTE else 0.0
        )
        return diverging_range(values, midpoint)

    def str_tokens(self) -> dict[str, list[str]]:
        prompts = [("target_", self.operation_input.target_tokens)]
        if self.operation_input.source_prompt is not None:
            prompts.append(("source_", self.operation_input.source_tokens))
        labels = {prefix: str_token_labels(self.model, tokens) for prefix, tokens in prompts}
        return {f"{prefix}{axis.value}": labels[prefix] for prefix in labels for axis in POSITION_AXES}

    def hover(self, panel: MetricPair, position: int, coordinate: str) -> str:
        row = self.result.iloc[position]
        target = describe_row(row, "target_")
        column = self.value_column(panel)
        lines = [f"<b>{column} = {format_value(row[column])}</b>"]
        if "str_generation" in row.index:
            lines.append(f'generation: "{format_value(row["str_generation"])}"')
        if coordinate and coordinate != target:
            lines.append(coordinate)
        lines.append(f"source: {describe_row(row, 'source_')}")
        lines.append(f"target: {target}")
        if self.per_row[position]:
            lines.append(self.per_row[position])
        return "<br>".join(line for line in lines if line)

    def _prompt(self, run: str, prompt: str, generation: int) -> str:
        return f'{run}: "{escape(prompt)}" generates "{format_value(str_token(self.model, generation))}"'

    def _companions(self) -> tuple[str, list[str]]:
        """Describe what the patch elements other than the laid-out one did.

        An element either has one row, the same location on every run, so it belongs in the subtitle,
        or as many rows as the result, in which case it co-varies and is reported per cell.

        Returns:
            A clause for the subtitle, and a line for each result row's hover.
        """
        rows = len(self.result)
        constant = []
        varying = []
        layout = self.operation_input.layout_index
        for index in self.operation_input.index:
            if index is layout:
                continue
            frame = index.to_df()
            if len(frame) == 1:
                constant.append(describe_row(frame.iloc[0], "target_"))
            elif len(frame) == rows:
                varying.append([describe_row(frame.iloc[position], "target_") for position in range(rows)])

        per_row = [
            "<br>".join(f"also: {element[position]}" for element in varying) if varying else ""
            for position in range(rows)
        ]
        return " · ".join(f"also patched: {text}" for text in constant), per_row
