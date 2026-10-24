from __future__ import annotations

import abc
import logging
import os
from typing import TYPE_CHECKING, Any, Generic, TypeVar, get_args, get_origin

import pandas as pd

from entoscope.model import Model
from entoscope.model.index import Index
from entoscope.operation.config import CONFIG_FILE
from entoscope.operation.input import OperationInput
from entoscope.util.data import create_if_not_exists

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from entoscope.operation.visualization import Figure

InputT = TypeVar("InputT", bound=OperationInput)
FigureT = TypeVar("FigureT", bound="Figure")

logger = logging.getLogger(__name__)


class AbstractCollector(abc.ABC, Generic[InputT]):
    """Accumulates one operation run's per-location results and writes them to ``save_dir``.

    A subclass owns the shape of a result (a DataFrame, a list of graphs, ...), which it allocates as it
    is constructed, fills one location's cell at a time through ``collect`` and exposes as ``result``.
    ``save`` creates ``save_dir`` and hands it to ``_save_result``, which owns everything the run leaves
    behind: its result, the index files beside it and the config it ran under.

    Every collector takes the model whether or not it reads one, so an operator constructs whichever
    ``collector_class`` it was given the same way, and swapping in a collector that labels the prompt's
    tokens (or reads ``cfg``) for one that does not is only a class name.

    Args:
        operation_input: The input the run iterates.
        model: The model the run reads.
        save_dir: The directory ``save`` writes into, or ``None`` to write nothing.
    """

    def __init__(self, operation_input: InputT, model: Model, save_dir: str | None = None):
        self.operation_input = operation_input
        self.model = model
        self.save_dir = save_dir

    @property
    @abc.abstractmethod
    def result(self) -> Any:
        """What the run collected so far."""
        raise NotImplementedError

    @abc.abstractmethod
    def collect(self, *args: Any, **kwargs: Any) -> None:
        """Take one result from the run's method, filed under the index position it belongs to.

        Args:
            *args: The result and its position, as each collector declares them.
            **kwargs: The same, by keyword.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _save_result(self, save_dir: str) -> None:
        """Write the collected result, its index files and the config into ``save_dir``.

        Args:
            save_dir: An existing directory.
        """
        raise NotImplementedError

    def output_files(self) -> list[str]:
        """List the file names ``save`` writes into ``save_dir``, which a plan reports as the run's output.

        Subclasses extend this to mirror their own ``_save_result``. It is deliberately not abstract:
        ``collector_class`` is an extension point, so a collector that declares no files keeps working
        and is reported as writing none. A plan asks a collector nothing was collected into, so an
        override may read the input but never the collected result.

        Returns:
            The file names, relative to ``save_dir``.
        """
        return []

    def save(self) -> None:
        """Write the run's files into ``save_dir``, created if missing, and log where they went.

        Writes nothing when ``save_dir`` is ``None``.
        """
        if self.save_dir is None:
            return

        create_if_not_exists(self.save_dir)
        self._save_result(self.save_dir)

        logger.info("Results saved to %s", self.save_dir)


class FrameCollector(AbstractCollector[InputT]):
    """A collector whose result is a frame with one row per row of the index the run swept.

    Cells arrive one at a time under the index position they fill and are glued back onto that
    index's own columns, so a subclass says only which indices the run has and what a cell holds. It
    writes ``result.csv``, an index CSV per index and ``config.json``.

    Args:
        operation_input: The input the run iterates.
        model: The model the run reads.
        save_dir: The directory ``save`` writes into, or ``None`` to write nothing.
    """

    def __init__(self, operation_input: InputT, model: Model, save_dir: str | None = None):
        super().__init__(operation_input, model, save_dir)
        self._cells: dict[int, dict[str, Any]] = {}

    @property
    def result(self) -> pd.DataFrame:
        """The index's columns followed by the collected cells, one row per index row."""
        rows = pd.DataFrame([self._cells[position] for position in range(len(self._index))])
        return pd.concat([self._index.to_df().reset_index(drop=True), rows], axis=1)

    @abc.abstractmethod
    def _indices(self) -> dict[str, Index]:
        """Name the run's indices.

        Reached from ``output_files`` too, so it may read the index-resolved input and no more.

        Returns:
            The run's indices, keyed by the CSV file name each is written to. The first one lays out the
            result.
        """
        raise NotImplementedError

    @property
    def _index(self) -> Index:
        return next(iter(self._indices().values()))

    def _collect_cell(self, index: int, cell: dict[str, Any]) -> None:
        self._cells[index] = cell

    def _save_result(self, save_dir: str) -> None:
        self.result.to_csv(os.path.join(save_dir, "result.csv"), index=False)
        for name, index in self._indices().items():
            index.to_csv(os.path.join(save_dir, name))
        self.operation_input.config.save(save_dir)

    def output_files(self) -> list[str]:
        return ["result.csv", *self._indices(), CONFIG_FILE, *super().output_files()]


class FigureMixin(Generic[FigureT]):
    """Names the figure a collector draws, read off the collector's own type argument.

    The figure is the last type parameter of whichever ``FigureMixin`` a collector names, so it is
    stated once, in the base list. An explicit assignment still wins, and an intermediate subclass
    that leaves the parameter free resolves nothing and inherits.

    Attributes:
        figure_class: The ``Figure`` the collector draws.
    """

    figure_class: type[FigureT]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for base in getattr(cls, "__orig_bases__", ()):
            origin = get_origin(base)
            figure = get_args(base)[-1] if isinstance(origin, type) and issubclass(origin, FigureMixin) else None
            if isinstance(figure, type) and "figure_class" not in cls.__dict__:
                cls.figure_class = figure


class VisualizationCollector(FrameCollector[InputT], FigureMixin[FigureT], abc.ABC):
    """A ``FrameCollector`` that also draws its frame as an interactive Plotly ``result.html``.

    Kept apart from the base collectors so that anything which only needs the CSVs builds no figure,
    and so visualization is opt-in by collector choice rather than by a config flag. Every
    ``GridFigure`` is built from the same four arguments and reads what it draws off them, so a
    subclass names its figure in its base list and nothing else. The page's plots are handed back as
    ``figures`` too, so a run that writes nothing can still be looked at.
    """

    @property
    def figures(self) -> dict[Any, go.Figure]:
        """The plots ``result.html`` holds, drawn from what the run collected so far.

        Keyed by the figure's panels, in page order. They are drawn afresh on every read.
        """
        return self._plots().figures()

    def _plots(self) -> FigureT:
        return self.figure_class(self.result, self.operation_input, self.model, self._index)

    def _save_result(self, save_dir: str) -> None:
        super()._save_result(save_dir)
        self._plots().render(os.path.join(save_dir, "result.html"))

    def output_files(self) -> list[str]:
        return ["result.html", *super().output_files()]
