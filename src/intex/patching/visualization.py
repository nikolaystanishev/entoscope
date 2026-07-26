"""Interactive Plotly rendering of a patching run.

The grid is the run's own index — layer down, position across for the canonical sweep — coloured by
the metric on a diverging scale centred on the value an ineffective patch would produce, so warm
cells are the locations that moved the model toward the source.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape

import pandas as pd

from intex.operation.visualization import DIVERGING, GridFigure, describe_row, diverging_range, format_value, titled
from intex.patching.config import PatchingMetric
from intex.patching.input import PatchingInput


class PatchingFigure(GridFigure):
    # The patched run is the target run, so the target's own coordinates label the axes; the source's
    # are the fallback for an index the target does not span.
    prefixes = ("target_", "source_")
    colorscale = DIVERGING

    def __init__(self, result: pd.DataFrame, patching_input: PatchingInput, tokens: dict[str, list[str]]) -> None:
        super().__init__(result, tokens)
        self.patching_input = patching_input
        self.constant, self.per_row = self._companions()

    @property
    def title(self) -> str:
        config = self.patching_input.config
        return titled(f"Patching — {config.metric.value}", config)

    @property
    def subtitle(self) -> str:
        parts = [
            f'source: "{escape(self.patching_input.source_prompt)}" →'
            + f' target: "{escape(self.patching_input.target_prompt)}"'
        ]
        if self.constant:
            parts.append(self.constant)
        return " · ".join(parts)

    @property
    def value_column(self) -> str:
        return self.patching_input.config.metric.value

    def value_range(self, values: Sequence[float]) -> tuple[float, float]:
        # An ineffective patch reproduces the target's own logit diff; the normalized metric and the
        # KL difference are both defined so that value is zero.
        midpoint = (
            float(self.patching_input.target_baseline)
            if self.patching_input.config.metric is PatchingMetric.NORMALIZED_LOGIT_DIFF
            else 0.0
        )
        return diverging_range(values, midpoint)

    def hover(self, position: int, coordinate: str) -> str:
        row = self.result.iloc[position]
        target = describe_row(row, "target_")
        lines = [f"<b>{self.value_column} = {format_value(row[self.value_column])}</b>"]
        if coordinate and coordinate != target:
            lines.append(coordinate)
        lines.append(f"source: {describe_row(row, 'source_')}")
        lines.append(f"target: {target}")
        if self.per_row[position]:
            lines.append(self.per_row[position])
        return "<br>".join(line for line in lines if line)

    def _companions(self) -> tuple[str, list[str]]:
        """What the other patch elements did: a constant clause for the subtitle, a line per result row.

        An element either has one row — the same location on every run, so it belongs in the subtitle —
        or as many rows as the result, in which case it co-varies and is reported per cell.
        """
        rows = len(self.result)
        constant = []
        varying = []
        for index in self.patching_input.index[1:]:
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
