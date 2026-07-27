from __future__ import annotations

import abc
import os
from typing import TYPE_CHECKING, Any, Generic, TypeVar, get_args, get_origin

import pandas as pd

from intex.model import Model
from intex.model.index import Index
from intex.operation.config import CONFIG_FILE
from intex.operation.input import OperationInput
from intex.util.data import create_if_not_exists

# Only needed to bound ``figure_class``: importing a figure here would put plotly behind every collector.
if TYPE_CHECKING:
    from intex.operation.visualization import Figure

InputT = TypeVar("InputT", bound=OperationInput)
FigureT = TypeVar("FigureT", bound="Figure")


class AbstractCollector(abc.ABC, Generic[InputT]):
    """Accumulates one operation run's per-location results and writes them to ``save_dir``.

    A subclass owns the shape of a result — a DataFrame, a tensor, ... — through ``prepare``
    (allocate it), ``collect`` (fill in a single location's cell) and ``result`` (expose it).
    ``save`` creates ``save_dir``, hands it to ``_save_result`` — which owns everything the run
    leaves behind: its result, the index files beside it and the config it ran under — and reports
    where that landed.

    Every collector takes the model whether or not it reads one, so an operator constructs whichever
    ``collector_class`` it was given the same way; swapping in a collector that labels the prompt's
    tokens (or reads ``cfg``) for one that does not is then only a class name.
    """

    def __init__(self, operation_input: InputT, model: Model, save_dir: str | None = None):
        self.operation_input = operation_input
        self.model = model
        self.save_dir = save_dir

    @property
    @abc.abstractmethod
    def result(self) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def prepare(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def collect(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _save_result(self, save_dir: str) -> None:
        """Persist the collected result and its index file(s) into ``save_dir``."""
        raise NotImplementedError

    def output_files(self) -> list[str]:
        """The file names ``save`` writes into ``save_dir`` — what a plan reports as the run's output.

        Subclasses extend this to mirror their own ``_save_result``. Deliberately not abstract:
        ``collector_class`` is an extension point, so a collector that declares no files keeps
        working and is reported as writing none. A plan asks an unprepared collector, so an override
        may read the input but never the collected result.
        """
        return []

    def save(self) -> None:
        if self.save_dir is None:
            return

        create_if_not_exists(self.save_dir)
        self._save_result(self.save_dir)

        print(f"Results saved to {self.save_dir}")


class FrameCollector(AbstractCollector[InputT]):
    """A collector whose result is a frame with one row per row of the index the run swept.

    Cells arrive one at a time under the index position they fill and are glued back onto that
    index's own columns, so a subclass says only which indices the run has and what a cell holds.
    """

    _cells: dict[int, dict[str, Any]] | None = None

    @property
    def result(self) -> pd.DataFrame:
        assert self._cells is not None
        rows = pd.DataFrame([self._cells[position] for position in range(len(self._index))])
        return pd.concat([self._index.to_df().reset_index(drop=True), rows], axis=1)

    def prepare(self) -> None:
        self._cells = {}

    @abc.abstractmethod
    def _indices(self) -> dict[str, Index]:
        """The run's indices, keyed by the CSV file name each is written to.

        Reached from ``output_files`` too, so it may read the (index-resolved) input and no more.
        """
        raise NotImplementedError

    @property
    def _index(self) -> Index:
        # The first index lays the result out; any others only record what else each run touched.
        return next(iter(self._indices().values()))

    def _collect_cell(self, index: int, cell: dict[str, Any]) -> None:
        assert self._cells is not None
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
    subclass names its figure in its base list and nothing else.
    """

    def _save_result(self, save_dir: str) -> None:
        super()._save_result(save_dir)
        self.figure_class(self.result, self.operation_input, self.model, self._index).render(
            os.path.join(save_dir, "result.html")
        )

    def output_files(self) -> list[str]:
        return ["result.html", *super().output_files()]
