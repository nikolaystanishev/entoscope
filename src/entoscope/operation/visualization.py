"""Index-grid plotting shared by the operations whose result is a tidy frame over an ``Index``.

An ``Index`` lays its rows out as dense per-axis codes and the result frame carries the original
``source_*``/``target_*`` values, so laying one out is the same problem for every operation and only
the cell payload differs. Everything produced here ends up inside plotly HTML, so values are escaped
once, in ``format_value``, rather than at each call site. ``Figure``, the page every operation's
HTML is written through, grid-drawn or not, lives here too.

A run rarely reads one number off itself: patching names several metrics and the lens keeps a whole
top-k. A page is therefore a list of panels, each drawn as its own plot from the same result, and an
operation says only what its panels are.
"""

from __future__ import annotations

import abc
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from html import escape
from typing import Any, Generic, TypeVar

import pandas as pd
import plotly.graph_objects as go

from entoscope.model import Model
from entoscope.model.config import ModelAxis
from entoscope.model.index import Index
from entoscope.operation.config import AbstractConfig
from entoscope.operation.input import OperationInput

InputT = TypeVar("InputT", bound=OperationInput)
PanelT = TypeVar("PanelT")

AXIS_ORDER: tuple[ModelAxis, ...] = (
    ModelAxis.LAYER,
    ModelAxis.HEAD,
    ModelAxis.NEURON,
    ModelAxis.POSITION,
    ModelAxis.DESTINATION_POSITION,
    ModelAxis.SOURCE_POSITION,
)

DIVERGING = [(0.0, "#2166ac"), (0.5, "#f7f7f7"), (1.0, "#b2182b")]
SEQUENTIAL = [(0.0, "#f7fbff"), (1.0, "#4292c6")]

MAX_ANNOTATED_CELLS = 600


def format_value(value: Any) -> str:
    """Format a result cell for a page, HTML-escaped.

    Args:
        value: An enum, a list or tuple, a float, ``None`` or anything ``str`` can print.

    Returns:
        An enum's value, a list in brackets, a float to four significant digits, ``"-"`` for a missing
        value, or ``str(value)``.
    """
    if isinstance(value, Enum):
        return escape(str(value.value))
    if isinstance(value, list | tuple):
        return escape(str(list(value)))
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if isinstance(value, float):
        return escape(f"{value:.4g}")
    return escape(str(value))


def label_column(frame: pd.DataFrame, axis: str, prefixes: Sequence[str]) -> str | None:
    """Find the ``source_*`` or ``target_*`` column holding the real values behind ``axis``'s dense codes.

    A column that is not fully populated is rejected: concatenating a shorter source index onto a
    longer target one leaves NaN behind, and a half-filled axis labels nothing.

    Args:
        frame: An index's frame, as ``Index.to_df`` lays it out.
        axis: An axis of the index, such as ``"layer"``.
        prefixes: The column prefixes to try, in order of preference.

    Returns:
        The first fully populated column, or ``None`` when there is none.
    """
    for prefix in prefixes:
        candidate = f"{prefix}{axis}"
        if candidate in frame.columns and frame[candidate].notna().all():
            return candidate
    return None


def tick_text(
    frame: pd.DataFrame, grid: IndexGrid, axis: str, prefixes: Sequence[str], tokens: dict[str, list[str]]
) -> dict[int, str]:
    """Label every code of one axis of the grid.

    A position is labeled with its token, a row without a value on the axis with its full coordinates.

    Args:
        frame: The index's own frame, one row per result row, as ``Index.to_df`` lays it out.
        grid: The grid.
        axis: The axis to label.
        prefixes: The column prefixes to read values from, in order of preference.
        tokens: The prompt's token strings, keyed by the position column they label.

    Returns:
        The label of each of the axis's codes.
    """
    labels: dict[int, str] = {}
    source = label_column(frame, axis, prefixes)
    column_tokens = tokens.get(source, []) if source is not None else []

    for position, code in enumerate(grid.codes[axis]):
        if code in labels:
            continue
        if source is None:
            labels[code] = describe_row(frame.iloc[position], prefixes[0])
            continue
        value = frame[source].iloc[position]
        if pd.api.types.is_integer(value) and 0 <= value < len(column_tokens):
            labels[code] = f'{value}: "{format_value(column_tokens[value])}"'
        else:
            labels[code] = format_value(value)

    return labels


def describe_row(row: pd.Series, prefix: str) -> str:
    """Describe the location of one index row on one side of the index.

    Args:
        row: A row of the index's frame, as ``Index.to_df`` lays it out.
        prefix: The side to describe, ``"source_"`` or ``"target_"``.

    Returns:
        The activation followed by every axis the row names, such as ``attn.z · layer 3 · head 5``.
    """
    columns = [column for column in row.index if column.startswith(prefix)]
    activation = f"{prefix}{ModelAxis.ACTIVATION.value}"
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
        parts.append(
            format_value(value) if column == activation else _named(column.removeprefix(prefix), format_value(value))
        )

    return " · ".join(parts)


def diverging_range(values: Sequence[float], midpoint: float) -> tuple[float, float]:
    """Centre a colour range on ``midpoint`` wide enough for every value.

    Args:
        values: The values to colour, NaN and infinities ignored, as no finite range holds them.
        midpoint: The value drawn in the middle colour.

    Returns:
        The range's low and high ends, one unit either side of ``midpoint`` when every value equals it.
    """
    spread = max((abs(value - midpoint) for value in values if math.isfinite(value)), default=0.0) or 1.0
    return midpoint - spread, midpoint + spread


def write_html(path: str, title: str, *figures: go.Figure) -> None:
    """Write figures one under the other into a self-contained HTML page.

    plotly.js is embedded once, with the first figure, so the page opens with no network.

    Args:
        path: The file to write.
        title: The page's title.
        *figures: The figures, in page order.
    """
    body = "<hr/>".join(figure.to_html(full_html=False, include_plotlyjs=(i == 0)) for i, figure in enumerate(figures))
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "<!doctype html><html><head><meta charset='utf-8'/>"
            f"<title>{escape(title)}</title></head><body>"
            f"{body}"
            "</body></html>"
        )


class Figure(abc.ABC, Generic[PanelT]):
    """A self-contained interactive HTML page, one plotly figure per panel.

    ``title`` names the page, and each panel titles its own plot, so the two do not have to repeat.
    """

    @property
    @abc.abstractmethod
    def title(self) -> str:
        """The page's title."""
        raise NotImplementedError

    @abc.abstractmethod
    def panels(self) -> list[PanelT]:
        """Name the page's panels.

        Returns:
            The panels, in page order, each drawn as its own plot.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def figure(self, panel: PanelT) -> go.Figure:
        """Draw one panel.

        Args:
            panel: One of ``panels()``.

        Returns:
            The panel's plot.
        """
        raise NotImplementedError

    def figures(self) -> dict[PanelT, go.Figure]:
        """Draw every panel.

        Returns:
            Each panel's plot, keyed by the panel, in page order.
        """
        return {panel: self.figure(panel) for panel in self.panels()}

    def render(self, path: str) -> None:
        """Write the page.

        Args:
            path: The HTML file to write.
        """
        write_html(path, self.title, *self.figures().values())


@dataclass(frozen=True)
class IndexGrid:
    """Where each result row sits once the axes a run varies over are laid out.

    Built from the index's own layout codes rather than from the result frame, so the axis rule can be
    read, and tested, without a run behind it. An axis the run does not vary over is dropped: it
    offers no comparison and belongs in the subtitle. Axes beyond the first two fold into composite y
    keys rather than being aggregated away, so nothing is hidden. ``y_keys`` ascends and plotly draws
    the first at the bottom, putting layer 0 there as the circuit figures do.

    Attributes:
        x_axis: The axis across, the last varying one in ``AXIS_ORDER``, or ``None`` when nothing varies.
        y_axes: The other varying axes, folded together into the rows.
        codes: Every axis's code for every result row.
        x_codes: The distinct codes across, ascending.
        y_keys: The distinct row keys, one code per ``y_axes`` entry, ascending.
        rows: The result row at each ``(y_key, x_code)`` cell.
    """

    x_axis: str | None
    y_axes: list[str]
    codes: dict[str, list[int]]
    x_codes: list[int]
    y_keys: list[tuple[int, ...]]
    rows: dict[tuple[tuple[int, ...], int], int]

    @classmethod
    def of(cls, layout: pd.DataFrame) -> IndexGrid:
        """Lay out an index.

        Args:
            layout: The index's layout codes, one column per axis, as ``Index.layout_df`` holds them.

        Returns:
            The grid.
        """
        axes = [str(column) for column in layout.columns]
        codes = {
            axis: [int(code) for code in layout[column]] for axis, column in zip(axes, layout.columns, strict=True)
        }
        varying = [axis for axis in axes if len(set(codes[axis])) > 1]
        ordered = sorted(varying, key=lambda axis: (_axis_rank(_location(axis)), axes.index(axis)))

        x_axis = ordered[-1] if ordered else None
        y_axes = ordered[:-1]

        x_values = codes[x_axis] if x_axis is not None else [0] * len(layout)
        y_values = list(zip(*(codes[axis] for axis in y_axes), strict=True)) if y_axes else [()] * len(layout)

        return cls(
            x_axis=x_axis,
            y_axes=y_axes,
            codes=codes,
            x_codes=sorted(set(x_values)),
            y_keys=sorted(set(y_values)),
            rows={cell: position for position, cell in enumerate(zip(y_values, x_values, strict=True))},
        )

    @property
    def is_bar(self) -> bool:
        """Whether at most one axis varies, so the grid is drawn as a bar chart."""
        return not self.y_axes

    def lay_out(self, values: Sequence[Any], fill: Any = None) -> list[list[Any]]:
        """Place one value per result row onto the grid.

        Args:
            values: One value per result row.
            fill: The value of a cell no row lands on.

        Returns:
            One list per row key, each holding one value per x code.
        """
        return [
            [values[self.rows[(key, code)]] if (key, code) in self.rows else fill for code in self.x_codes]
            for key in self.y_keys
        ]


def x_labels(frame: pd.DataFrame, grid: IndexGrid, prefixes: Sequence[str], tokens: dict[str, list[str]]) -> list[str]:
    """Label the grid's columns.

    Args:
        frame: The index's own frame, one row per result row, as ``Index.to_df`` lays it out.
        grid: The grid.
        prefixes: The column prefixes to read values from, in order of preference.
        tokens: The prompt's token strings, keyed by the position column they label.

    Returns:
        One label per x code, or the single row's location when no axis varies.
    """
    if grid.x_axis is None:
        return [describe_row(frame.iloc[0], prefixes[0])] if len(frame) else []
    labels = tick_text(frame, grid, grid.x_axis, prefixes, tokens)
    return [labels.get(code, str(code)) for code in grid.x_codes]


def y_labels(frame: pd.DataFrame, grid: IndexGrid, prefixes: Sequence[str], tokens: dict[str, list[str]]) -> list[str]:
    """Label the grid's rows.

    Args:
        frame: The index's own frame, one row per result row, as ``Index.to_df`` lays it out.
        grid: The grid.
        prefixes: The column prefixes to read values from, in order of preference.
        tokens: The prompt's token strings, keyed by the position column they label.

    Returns:
        One label per row key, each axis named when several fold into the rows.
    """
    per_axis = [tick_text(frame, grid, axis, prefixes, tokens) for axis in grid.y_axes]
    named = len(grid.y_axes) > 1
    return [
        " · ".join(
            _named(axis, labels.get(code, str(code))) if named else labels.get(code, str(code))
            for axis, labels, code in zip(grid.y_axes, per_axis, key, strict=True)
        )
        for key in grid.y_keys
    ]


def cell_labels(
    frame: pd.DataFrame, grid: IndexGrid, prefixes: Sequence[str], tokens: dict[str, list[str]]
) -> list[str]:
    """Give every result row the coordinates of its cell.

    Args:
        frame: The index's own frame, one row per result row, as ``Index.to_df`` lays it out.
        grid: The grid.
        prefixes: The column prefixes to read values from, in order of preference.
        tokens: The prompt's token strings, keyed by the position column they label.

    Returns:
        One label per result row, such as ``layer 3 · position 4: " France"``.
    """
    axes = [*grid.y_axes, *([grid.x_axis] if grid.x_axis is not None else [])]
    per_axis = [tick_text(frame, grid, axis, prefixes, tokens) for axis in axes]

    labels = []
    for position in range(len(frame)):
        parts = [
            _named(axis, axis_labels.get(grid.codes[axis][position], ""))
            for axis, axis_labels in zip(axes, per_axis, strict=True)
        ]
        labels.append(" · ".join(part for part in parts if part))

    return labels


def axis_title(axes: Sequence[str]) -> str:
    return " · ".join(axes)


def titled(kind: str, config: AbstractConfig) -> str:
    """Title a page after the operation and, when it has one, the config's id.

    Args:
        kind: What the page shows, such as ``"Logit lens"``.
        config: The config the run was made under.

    Returns:
        ``kind``, followed by the config's id in parentheses when it has one.
    """
    return f"{kind} (config: {config.config_id})" if config.config_id else kind


class GridFigure(Figure[PanelT], Generic[InputT, PanelT]):
    """An operation's result frame drawn as the grid its index describes, once per panel.

    Choosing the axes, pivoting the rows onto them and falling back to a bar when only one of them
    varies is the same work whatever was run, so it lives here. A subclass says only what its own
    result means: which column a panel holds its value in, how to colour it, and what a cell reads on
    hover. Those answers are read off the run's input, which comes along for that reason, and because
    it lets a collector name a figure class and construct it without knowing what produced the frame.
    Every panel shares one grid and one set of axis labels, so those are resolved once, at
    construction.

    Args:
        result: The run's result frame.
        operation_input: The input the run iterated.
        model: The model the run read.
        index: The index that laid the result out, whose layout codes build the grid.

    Attributes:
        coordinates: The index's own frame, which every cell, tick and hover is labelled from: the result
            frame also holds the values, and a value column named with a side's prefix would read as a
            coordinate.
        prefixes: The column prefixes axis values are read from, in order of preference: the index's own
            ``layout_prefixes``, so the axes are labelled by the side the grid was laid out along.
        colorscale: The plotly colour scale cells are drawn with.
    """

    colorscale = SEQUENTIAL

    def __init__(self, result: pd.DataFrame, operation_input: InputT, model: Model, index: Index) -> None:
        self.result = result
        self.operation_input = operation_input
        self.model = model
        self.coordinates = index.to_df()
        self.prefixes = index.layout_prefixes
        self.tokens = self.str_tokens()
        self.grid = IndexGrid.of(index.layout_df)

    @abc.abstractmethod
    def heading(self, panel: PanelT) -> str:
        """Name one panel's plot.

        Args:
            panel: One of ``panels()``.

        Returns:
            The plot's heading, shown above the subtitle line.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def subtitle(self, panel: PanelT) -> str:
        """Describe what one panel's plot holds, under its heading.

        Args:
            panel: One of ``panels()``.

        Returns:
            The subtitle, as HTML.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def value_column(self, panel: PanelT) -> str:
        """Name the result column one panel colours its cells by.

        Args:
            panel: One of ``panels()``.

        Returns:
            A column of the result frame.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def hover(self, panel: PanelT, position: int, coordinate: str) -> str:
        """Write what one cell shows on hover.

        Args:
            panel: One of ``panels()``.
            position: The result row the cell is built from.
            coordinate: The cell's laid-out coordinates, from ``cell_labels``.

        Returns:
            The hover text, as HTML.
        """
        raise NotImplementedError

    def value_title(self, panel: PanelT) -> str:
        """Title one panel's colour bar.

        Args:
            panel: One of ``panels()``.

        Returns:
            The title, as text, by default the value column's name.
        """
        return self.value_column(panel)

    def value_range(self, panel: PanelT, values: Sequence[float]) -> tuple[float, float]:
        """Choose the range one panel's colours span.

        Args:
            panel: One of ``panels()``.
            values: The panel's values, one per result row.

        Returns:
            The low and high ends of the colour range, by default 0 and 1.
        """
        return 0.0, 1.0

    def bar_yaxis(self, panel: PanelT) -> dict:
        """Lay out the value axis of one panel drawn as a bar chart.

        Args:
            panel: One of ``panels()``.

        Returns:
            plotly ``yaxis`` settings, by default only the axis title.
        """
        return dict(title=escape(self.value_title(panel)))

    def cell_text(self, panel: PanelT) -> list[str] | None:
        """Choose the label printed inside each cell of one panel.

        The size cap is applied by ``figure``, so a subclass answers for its own result alone.

        Args:
            panel: One of ``panels()``.

        Returns:
            One label per result row, or ``None`` to leave the cells bare, the default.
        """
        return None

    def str_tokens(self) -> dict[str, list[str]]:
        """Read the prompt's token strings, to label the position axes with.

        Read once at construction. A figure whose index carries no position, or a run with no prompt to
        read one from, labels its ticks with the coordinates alone.

        Returns:
            The token strings for each position column of the result, keyed by that column. Empty by
            default.
        """
        return {}

    def figures(self) -> dict[PanelT, go.Figure]:
        """Draw every panel, or only the first, saying so, when the index has no rows.

        Returns:
            Each panel's plot, keyed by the panel, in page order.
        """
        panels = self.panels()
        if self.result.empty:
            return {panels[0]: self.figure(panels[0])} if panels else {}
        return super().figures()

    @property
    def _x_positions(self) -> list[int]:
        return list(range(len(self.grid.x_codes)))

    @property
    def _y_positions(self) -> list[int]:
        return list(range(len(self.grid.y_keys)))

    def figure(self, panel: PanelT) -> go.Figure:
        """Draw one panel as a heatmap over the grid, or a bar chart when at most one axis varies.

        Args:
            panel: One of ``panels()``.

        Returns:
            The panel's plot.
        """
        layout = dict(
            title=f"{escape(self.heading(panel))}<br><sub>{self.subtitle(panel)}</sub>",
            template="plotly_white",
            margin=dict(l=90, r=40, t=100, b=80),
        )

        if self.result.empty or self.value_column(panel) not in self.result.columns:
            figure = go.Figure()
            figure.add_annotation(text="No rows in the index", showarrow=False, font=dict(size=16))
            figure.update_layout(**layout)
            return figure

        grid = self.grid
        values = [float(value) for value in self.result[self.value_column(panel)]]
        low, high = self.value_range(panel, values)
        coordinates = cell_labels(self.coordinates, grid, self.prefixes, self.tokens)
        hover = [self.hover(panel, position, coordinates[position]) for position in range(len(self.result))]
        text = self.cell_text(panel) if len(grid.x_codes) * len(grid.y_keys) <= MAX_ANNOTATED_CELLS else None

        draw = self._bar if grid.is_bar else self._heatmap
        figure = draw(panel, values, hover, text, low, high)
        figure.update_layout(
            **layout,
            width=min(2000, max(640, 90 * len(grid.x_codes) + 260)),
            height=min(2400, max(360, 26 * len(grid.y_keys) + 220)),
            xaxis=dict(
                title=axis_title([grid.x_axis]) if grid.x_axis is not None else "",
                tickmode="array",
                tickvals=self._x_positions,
                ticktext=x_labels(self.coordinates, grid, self.prefixes, self.tokens),
            ),
        )
        return figure

    def _bar(
        self,
        panel: PanelT,
        values: list[float],
        hover: list[str],
        text: list[str] | None,
        low: float,
        high: float,
    ) -> go.Figure:
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
                    colorbar=dict(title=escape(self.value_title(panel))),
                ),
            )
        )
        figure.update_layout(yaxis=self.bar_yaxis(panel))
        return figure

    def _heatmap(
        self,
        panel: PanelT,
        values: list[float],
        hover: list[str],
        text: list[str] | None,
        low: float,
        high: float,
    ) -> go.Figure:
        heatmap = go.Heatmap(
            z=self.grid.lay_out(values),
            x=self._x_positions,
            y=self._y_positions,
            colorscale=self.colorscale,
            zmin=low,
            zmax=high,
            colorbar=dict(title=escape(self.value_title(panel))),
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
                title=axis_title(self.grid.y_axes),
                tickmode="array",
                tickvals=self._y_positions,
                ticktext=y_labels(self.coordinates, self.grid, self.prefixes, self.tokens),
            )
        )
        return figure


def _location(name: str) -> ModelAxis | None:
    try:
        return ModelAxis(name)
    except ValueError:
        return None


def _axis_rank(location: ModelAxis | None) -> int:
    return AXIS_ORDER.index(location) if location in AXIS_ORDER else len(AXIS_ORDER)


def _named(axis: str, label: str) -> str:
    return f"{axis} {label}" if _location(axis) is not None else label
