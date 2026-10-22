"""Interactive Plotly rendering of a patching run.

The grid is the run's own index, layer up and position across for the canonical sweep, coloured by
the metric on a diverging scale centred on the value an ineffective patch would produce, so warm
cells are the locations that moved the model toward the source. One such grid is drawn per column
the config names, each centred on its own comparison's ineffective value. A cell's hover names the
token its patched run predicts next, and the subtitle the token each unpatched prompt does and the
tokens the grid's metric reads.
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
from entoscope.patching.config import MetricComparison
from entoscope.patching.input import PatchingInput


def metric_column(metric: str) -> str:
    """Name the result column of a metric.

    Every metric's column carries the same suffix, so none is a column the run writes itself: ``generation``,
    ``str_generation`` or one of the index's.

    Args:
        metric: The metric's name in the config.

    Returns:
        ``<name>_metric``, such as ``logit_diff_metric``.
    """
    return f"{metric}_metric"


class PatchingFigure(GridFigure[PatchingInput, str]):
    """The page of a patching run: one grid per result column, each panel a metric, by its name.

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
        self.coordinates = index.to_df()
        self.constant, self.per_row = self._companions()

    @property
    def title(self) -> str:
        return titled("Patching", self.operation_input.config)

    def panels(self) -> list[str]:
        return list(self.operation_input.config.metric)

    def heading(self, panel: str) -> str:
        config = self.operation_input.config
        return titled(f"Patching - {config.method.type} · {self.value_column(panel)}", config)

    def subtitle(self, panel: str) -> str:
        patching_input = self.operation_input
        prompts = self._prompt("target", patching_input.target_prompt, patching_input.target_generation)
        if patching_input.source_prompt is not None:
            source = self._prompt("source", patching_input.source_prompt, patching_input.source_generation)
            prompts = f"{source} → {prompts}"
        parts = [prompts, self._reads(panel)]
        if self.constant:
            parts.append(self.constant)
        return " · ".join(parts)

    def value_column(self, panel: str) -> str:
        return metric_column(panel)

    def value_range(self, panel: str, values: Sequence[float]) -> tuple[float, float]:
        absolute = self.operation_input.config.metric[panel].comparison is MetricComparison.ABSOLUTE
        midpoint = float(self.operation_input.metrics[panel].target(self.operation_input)) if absolute else 0.0
        return diverging_range(values, midpoint)

    def str_tokens(self) -> dict[str, list[str]]:
        prompts = [("target_", self.operation_input.target_tokens)]
        if self.operation_input.source_prompt is not None:
            prompts.append(("source_", self.operation_input.source_tokens))
        labels = {prefix: str_token_labels(self.model, tokens) for prefix, tokens in prompts}
        return {f"{prefix}{axis.value}": labels[prefix] for prefix in labels for axis in POSITION_AXES}

    def hover(self, panel: str, position: int, coordinate: str) -> str:
        row = self.result.iloc[position]
        # Described off the index's own columns, since the result's also hold the metrics, and a metric
        # named with a side's prefix would read as a coordinate.
        coordinates = self.coordinates.iloc[position]
        target = describe_row(coordinates, "target_")
        column = self.value_column(panel)
        lines = [f"<b>{column} = {format_value(row[column])}</b>"]
        if "str_generation" in row.index:
            lines.append(f'generation: "{format_value(row["str_generation"])}"')
        if coordinate and coordinate != target:
            lines.append(coordinate)
        lines.append(f"source: {describe_row(coordinates, 'source_')}")
        lines.append(f"target: {target}")
        if self.per_row[position]:
            lines.append(self.per_row[position])
        return "<br>".join(line for line in lines if line)

    def _prompt(self, run: str, prompt: str, generation: int) -> str:
        return f'{run}: "{escape(prompt)}" generates "{format_value(str_token(self.model, generation))}"'

    def _reads(self, metric: str) -> str:
        """Spell out what a metric measures, since its name is whatever the config chose.

        Args:
            metric: The metric's name in the config.

        Returns:
            The metric's type, then every token it reads, by name and spelled, such as
            ``logit_diff: correct " Paris", distractor " Madrid"``.
        """
        config = self.operation_input.config.metric[metric]
        tokens = [
            f'{escape(token)} "{format_value(str_token(self.model, token_id))}"'
            for token, token_id in zip(config.tokens, self.operation_input.metrics[metric].tokens, strict=True)
        ]
        return f"{config.type.value}: {', '.join(tokens)}" if tokens else config.type.value

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
