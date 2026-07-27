"""Index-grid plotting shared by the operations whose result is a tidy frame over an ``Index``.

A result frame carries dense ``index_*`` codes alongside the original ``source_*``/``target_*``
values, so laying one out is the same problem for every operation and only the cell payload differs.
Everything produced here ends up inside plotly HTML, so values are escaped once, in ``format_value``,
rather than at each call site. ``Figure`` — the page every operation's HTML is written through,
grid-drawn or not — lives here too.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from html import escape
from typing import Any, Generic, TypeVar

import pandas as pd
import plotly.graph_objects as go

from intex.model import Model
from intex.model.config import ModelLocation
from intex.operation.config import AbstractConfig
from intex.operation.input import OperationInput

InputT = TypeVar("InputT", bound=OperationInput)

# x takes the last of these a run varies over and y the rest, so an activation-patching grid comes
# out position-across / layer-down and a layer x head or layer x neuron grid keeps layer down too.
AXIS_ORDER: tuple[ModelLocation, ...] = (
    ModelLocation.LAYER,
    ModelLocation.HEAD,
    ModelLocation.NEURON,
    ModelLocation.POSITION,
)

INDEX_PREFIX = "index_"

DIVERGING = [(0.0, "#2166ac"), (0.5, "#f7f7f7"), (1.0, "#b2182b")]
# Stops well short of navy: a heatmap's cell text takes a single colour for the whole trace, so the
# top of the scale has to stay light enough to read the lens's top-1 token against.
SEQUENTIAL = [(0.0, "#f7fbff"), (1.0, "#4292c6")]

# Past this a per-cell label is unreadable anyway, and printing it only bloats the page.
MAX_ANNOTATED_CELLS = 600


def format_value(value: Any) -> str:
    if isinstance(value, Enum):
        return escape(str(value.value))
    if isinstance(value, list | tuple):
        return escape(str(list(value)))
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if isinstance(value, float):
        return escape(f"{value:.4g}")
    return escape(str(value))


def location_of(column: str) -> ModelLocation | None:
    return _location(column.removeprefix(INDEX_PREFIX))


def label_column(frame: pd.DataFrame, column: str, prefixes: Sequence[str]) -> str | None:
    """The ``source_*``/``target_*`` column holding the real values behind ``column``'s dense codes.

    Prefixes are tried in the caller's order of preference, and a column that is not fully populated
    is rejected: concatenating a shorter source index onto a longer target one leaves NaN behind, and
    a half-filled axis labels nothing.
    """
    for prefix in prefixes:
        candidate = f"{prefix}{column.removeprefix(INDEX_PREFIX)}"
        if candidate in frame.columns and frame[candidate].notna().all():
            return candidate
    return None


def tick_text(
    frame: pd.DataFrame, column: str, prefixes: Sequence[str], tokens: dict[str, list[str]]
) -> dict[int, str]:
    labels: dict[int, str] = {}
    source = label_column(frame, column, prefixes)
    column_tokens = tokens.get(source, []) if source is not None else []

    for position, code in enumerate(frame[column]):
        code = int(code)
        if code in labels:
            continue
        if source is None:
            labels[code] = describe_row(frame.iloc[position], prefixes[0])
            continue
        value = frame[source].iloc[position]
        if pd.api.types.is_integer(value) and 0 <= value < len(column_tokens):
            labels[code] = f'{value}: "{column_tokens[value]}"'
        else:
            labels[code] = format_value(value)

    return labels


def describe_row(row: pd.Series, prefix: str) -> str:
    columns = [column for column in row.index if column.startswith(prefix)]
    activation = f"{prefix}{ModelLocation.ACTIVATION.value}"
    ordered = [activation] if activation in columns else []
    ordered += sorted(
        (column for column in columns if column != activation),
        key=lambda column: (_axis_rank(_location(column.removeprefix(prefix))), column),
    )

    parts = []
    for column in ordered:
        value = row[column]
        if not isinstance(value, list | tuple) and pd.isna(value):
            continue
        parts.append(format_value(value) if column == activation else _named(column, prefix, format_value(value)))

    return " · ".join(parts)


def diverging_range(values: Sequence[float], midpoint: float) -> tuple[float, float]:
    spread = max((abs(value - midpoint) for value in values if not pd.isna(value)), default=0.0) or 1.0
    return midpoint - spread, midpoint + spread


def write_html(path: str, title: str, *figures: go.Figure) -> None:
    # plotly.js is embedded by the first figure and shared by the rest of the page.
    body = "<hr/>".join(figure.to_html(full_html=False, include_plotlyjs=(i == 0)) for i, figure in enumerate(figures))
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "<!doctype html><html><head><meta charset='utf-8'/>"
            f"<title>{escape(title)}</title></head><body>"
            f"{body}"
            "</body></html>"
        )


class Figure(abc.ABC):
    """A self-contained interactive HTML page built from one or more plotly figures."""

    @property
    @abc.abstractmethod
    def title(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def figures(self) -> list[go.Figure]:
        raise NotImplementedError

    def render(self, path: str) -> None:
        write_html(path, self.title, *self.figures())


@dataclass(frozen=True)
class IndexGrid:
    """Where each result row sits once the axes a run varies over are laid out.

    Pure frame logic with no dependency on a live model, so the axis rule can be read — and tested —
    on its own. An axis the run does not vary over is dropped: it offers no comparison and belongs in
    the subtitle. Axes beyond the first two fold into composite y keys rather than being aggregated
    away, so nothing is hidden; ``y_keys`` ascends and plotly draws the first at the bottom, putting
    layer 0 there as the circuit figures do.
    """

    x_column: str | None
    y_columns: list[str]
    x_codes: list[int]
    y_keys: list[tuple[int, ...]]
    rows: dict[tuple[tuple[int, ...], int], int]

    @classmethod
    def of(cls, frame: pd.DataFrame) -> IndexGrid:
        columns = [column for column in frame.columns if column.startswith(INDEX_PREFIX)]
        varying = [column for column in columns if frame[column].nunique(dropna=False) > 1]
        ordered = sorted(varying, key=lambda column: (_axis_rank(location_of(column)), columns.index(column)))

        x_column = ordered[-1] if ordered else None
        y_columns = ordered[:-1]

        x_values = [int(code) for code in frame[x_column]] if x_column is not None else [0] * len(frame)
        y_values = (
            [tuple(int(code) for code in key) for key in zip(*(frame[column] for column in y_columns), strict=True)]
            if y_columns
            else [()] * len(frame)
        )

        return cls(
            x_column=x_column,
            y_columns=y_columns,
            x_codes=sorted(set(x_values)),
            y_keys=sorted(set(y_values)),
            rows={cell: position for position, cell in enumerate(zip(y_values, x_values, strict=True))},
        )

    @property
    def is_bar(self) -> bool:
        return not self.y_columns

    def lay_out(self, values: Sequence[Any], fill: Any = None) -> list[list[Any]]:
        return [
            [values[self.rows[(key, code)]] if (key, code) in self.rows else fill for code in self.x_codes]
            for key in self.y_keys
        ]


def x_labels(frame: pd.DataFrame, grid: IndexGrid, prefixes: Sequence[str], tokens: dict[str, list[str]]) -> list[str]:
    if grid.x_column is None:
        return [describe_row(frame.iloc[0], prefixes[0])] if len(frame) else []
    labels = tick_text(frame, grid.x_column, prefixes, tokens)
    return [labels.get(code, str(code)) for code in grid.x_codes]


def y_labels(frame: pd.DataFrame, grid: IndexGrid, prefixes: Sequence[str], tokens: dict[str, list[str]]) -> list[str]:
    per_column = [tick_text(frame, column, prefixes, tokens) for column in grid.y_columns]
    named = len(grid.y_columns) > 1
    return [
        " · ".join(
            _named(column, INDEX_PREFIX, labels.get(code, str(code))) if named else labels.get(code, str(code))
            for column, labels, code in zip(grid.y_columns, per_column, key, strict=True)
        )
        for key in grid.y_keys
    ]


def cell_labels(
    frame: pd.DataFrame, grid: IndexGrid, prefixes: Sequence[str], tokens: dict[str, list[str]]
) -> list[str]:
    columns = [*grid.y_columns, *([grid.x_column] if grid.x_column is not None else [])]
    per_column = [tick_text(frame, column, prefixes, tokens) for column in columns]

    labels = []
    for position in range(len(frame)):
        parts = [
            _named(column, INDEX_PREFIX, column_labels.get(int(frame[column].iloc[position]), ""))
            for column, column_labels in zip(columns, per_column, strict=True)
        ]
        labels.append(" · ".join(part for part in parts if part))

    return labels


def axis_title(columns: Sequence[str]) -> str:
    return " · ".join(column.removeprefix(INDEX_PREFIX) for column in columns)


def titled(kind: str, config: AbstractConfig) -> str:
    return f"{kind} (config: {config.config_id})" if config.config_id else kind


class GridFigure(Figure, Generic[InputT]):
    """An operation's result frame drawn as the grid its index describes.

    Choosing the axes, pivoting the rows onto them and falling back to a bar when only one of them
    varies is the same work whatever was run, so it lives here; a subclass says only what its own
    result means — which column holds the value, how to colour it, and what a cell reads on hover.
    Those answers are read off the run's input, which comes along for that reason — and because it
    lets a collector name a figure class and construct it without knowing what produced the frame.
    """

    prefixes: tuple[str, ...] = ("source_",)
    colorscale = SEQUENTIAL

    def __init__(self, result: pd.DataFrame, operation_input: InputT, model: Model) -> None:
        self.result = result
        self.operation_input = operation_input
        self.model = model
        self.tokens = self.str_tokens()
        self.grid = IndexGrid.of(result)

    @property
    @abc.abstractmethod
    def subtitle(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def value_column(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def hover(self, position: int, coordinate: str) -> str:
        """What the cell built from result row ``position`` reads, given its laid-out coordinate."""
        raise NotImplementedError

    @property
    def value_title(self) -> str:
        return self.value_column

    def value_range(self, values: Sequence[float]) -> tuple[float, float]:
        return 0.0, 1.0

    def bar_yaxis(self) -> dict:
        return dict(title=self.value_title)

    def cell_text(self) -> list[str] | None:
        """One label per result row to print inside its cell, or ``None`` to leave the cells bare.

        The size cap is applied by ``figure``, so a subclass answers for its own result alone.
        """
        return None

    def str_tokens(self) -> dict[str, list[str]]:
        """The prompt's token strings for each position column of the result, keyed by that column.

        Read once at construction to label the axes; a figure whose index carries no position — or a
        run with no prompt to read one from — labels its ticks with the coordinates alone.
        """
        return {}

    def figures(self) -> list[go.Figure]:
        return [self.figure()]

    # Both axes are drawn against dense positions with the real values as tick text, so a duplicate
    # or non-printable label cannot merge two cells the way a categorical axis would.
    @property
    def _x_positions(self) -> list[int]:
        return list(range(len(self.grid.x_codes)))

    @property
    def _y_positions(self) -> list[int]:
        return list(range(len(self.grid.y_keys)))

    def figure(self) -> go.Figure:
        layout = dict(
            title=f"{escape(self.title)}<br><sub>{self.subtitle}</sub>",
            template="plotly_white",
            margin=dict(l=90, r=40, t=100, b=80),
        )

        # An empty index drops the value column entirely rather than leaving it empty.
        if self.result.empty or self.value_column not in self.result.columns:
            figure = go.Figure()
            figure.add_annotation(text="No rows in the index", showarrow=False, font=dict(size=16))
            figure.update_layout(**layout)
            return figure

        grid = self.grid
        values = [float(value) for value in self.result[self.value_column]]
        low, high = self.value_range(values)
        coordinates = cell_labels(self.result, grid, self.prefixes, self.tokens)
        hover = [self.hover(position, coordinates[position]) for position in range(len(self.result))]
        text = self.cell_text() if len(grid.x_codes) * len(grid.y_keys) <= MAX_ANNOTATED_CELLS else None

        draw = self._bar if grid.is_bar else self._heatmap
        figure = draw(values, hover, text, low, high)
        figure.update_layout(
            **layout,
            width=min(2000, max(640, 90 * len(grid.x_codes) + 260)),
            height=min(2400, max(360, 26 * len(grid.y_keys) + 220)),
            xaxis=dict(
                title=axis_title([grid.x_column]) if grid.x_column is not None else "",
                tickmode="array",
                tickvals=self._x_positions,
                ticktext=x_labels(self.result, grid, self.prefixes, self.tokens),
            ),
        )
        return figure

    def _bar(self, values: list[float], hover: list[str], text: list[str] | None, low: float, high: float) -> go.Figure:
        laid_out = self.grid.lay_out(values, 0.0)[0]
        figure = go.Figure(
            go.Bar(
                x=self._x_positions,
                y=laid_out,
                text=self.grid.lay_out(text, "")[0] if text is not None else None,
                textposition="outside" if text is not None else None,
                hovertext=self.grid.lay_out(hover, "")[0],
                hoverinfo="text",
                marker=dict(
                    color=laid_out,
                    colorscale=self.colorscale,
                    cmin=low,
                    cmax=high,
                    colorbar=dict(title=self.value_title),
                ),
            )
        )
        figure.update_layout(yaxis=self.bar_yaxis())
        return figure

    def _heatmap(
        self, values: list[float], hover: list[str], text: list[str] | None, low: float, high: float
    ) -> go.Figure:
        heatmap = go.Heatmap(
            z=self.grid.lay_out(values),
            x=self._x_positions,
            y=self._y_positions,
            colorscale=self.colorscale,
            zmin=low,
            zmax=high,
            colorbar=dict(title=self.value_title),
            hovertext=self.grid.lay_out(hover, ""),
            hoverinfo="text",
            xgap=1,
            ygap=1,
        )
        if text is not None:
            heatmap.text = self.grid.lay_out(text, "")
            heatmap.texttemplate = "%{text}"
            heatmap.textfont = dict(color="#111827", size=10)

        figure = go.Figure(heatmap)
        figure.update_layout(
            yaxis=dict(
                title=axis_title(self.grid.y_columns),
                tickmode="array",
                tickvals=self._y_positions,
                ticktext=y_labels(self.result, self.grid, self.prefixes, self.tokens),
            )
        )
        return figure


def _location(name: str) -> ModelLocation | None:
    try:
        return ModelLocation(name)
    except ValueError:
        return None


def _axis_rank(location: ModelLocation | None) -> int:
    return AXIS_ORDER.index(location) if location in AXIS_ORDER else len(AXIS_ORDER)


def _named(column: str, prefix: str, label: str) -> str:
    name = column.removeprefix(prefix)
    return f"{name} {label}" if _location(name) is not None else label
