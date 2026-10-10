"""Interactive Plotly rendering of a patching run.

The grid is the run's own index — layer down, position across for the canonical sweep — coloured by
the metric on a diverging scale centred on the value an ineffective patch would produce, so warm
cells are the locations that moved the model toward the source. One such grid is drawn per column
the config names, each centred on its own comparison's ineffective value. A cell's hover names the
token its patched run predicts next, and the subtitle the token each unpatched prompt does.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape

import pandas as pd

from intex.model import Model
from intex.model.index import Index
from intex.model.token import str_token, str_token_labels
from intex.operation.visualization import DIVERGING, GridFigure, describe_row, diverging_range, format_value, titled
from intex.patching.config import MetricComparison, MetricPair, metric_column
from intex.patching.input import PatchingInput


class PatchingFigure(GridFigure[PatchingInput, MetricPair]):
    # The patched run is the target run, so the target's own coordinates label the axes; the source's
    # are the fallback for an index the target does not span.
    prefixes = ("target_", "source_")
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
        # An ineffective patch reproduces the target run, so that is where the diverging scale is
        # centred: at the target's own metric when it is reported absolutely, and at zero once the
        # comparison has already subtracted it.
        metric, comparison = panel
        midpoint = (
            float(self.operation_input.target_baseline(metric)) if comparison is MetricComparison.ABSOLUTE else 0.0
        )
        return diverging_range(values, midpoint)

    def str_tokens(self) -> dict[str, list[str]]:
        prompts = [("target_", self.operation_input.target_tokens)]
        if self.operation_input.source_prompt is not None:
            prompts.append(("source_", self.operation_input.source_tokens))
        return {f"{prefix}position": str_token_labels(self.model, tokens) for prefix, tokens in prompts}

    def hover(self, panel: MetricPair, position: int, coordinate: str) -> str:
        row = self.result.iloc[position]
        target = describe_row(row, "target_")
        column = self.value_column(panel)
        lines = [f"<b>{column} = {format_value(row[column])}</b>"]
        # Only a method that runs the patched model has a generation to report.
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
        """What the other patch elements did: a constant clause for the subtitle, a line per result row.

        An element either has one row — the same location on every run, so it belongs in the subtitle —
        or as many rows as the result, in which case it co-varies and is reported per cell.
        """
        rows = len(self.result)
        constant = []
        varying = []
        for index in self.operation_input.index[1:]:
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
