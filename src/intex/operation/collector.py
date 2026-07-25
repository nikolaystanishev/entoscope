from __future__ import annotations

import abc
import os
from typing import Any, Generic, TypeVar

from intex.operation.input import OperationInput
from intex.util.collections import to_dict
from intex.util.data import create_if_not_exists, write_json

InputT = TypeVar("InputT", bound=OperationInput)


class AbstractCollector(abc.ABC, Generic[InputT]):
    """Accumulates one operation run's per-location results and writes them to ``save_dir``.

    A subclass owns the shape of a result — a DataFrame, a tensor, ... — through ``prepare``
    (allocate it), ``collect`` (fill in a single location's cell) and ``result`` (expose it).
    ``save`` is shared: it creates ``save_dir``, delegates the result and index files to
    ``_save_result``, dumps ``config.json`` and prints a completion line, so every operation
    lays its output out the same way.
    """

    def __init__(self, operation_input: InputT, save_dir: str | None = None):
        self.operation_input = operation_input
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
        ``collector_class`` is an extension point, so a collector that does not declare its files
        keeps working and is reported as writing the shared ``config.json`` alone.
        """
        return ["config.json"]

    def save(self) -> None:
        if self.save_dir is None:
            return

        create_if_not_exists(self.save_dir)
        self._save_result(self.save_dir)
        write_json(os.path.join(self.save_dir, "config.json"), to_dict(self.operation_input.config))

        print(f"{self.operation_input.config.config_id} done - results saved to {self.save_dir}")
