import abc
from collections.abc import Iterator
from typing import Any

from intex.dataset.config import DatasetConfig

Row = dict[str, Any]


class AbstractDataset(abc.ABC):
    """A collection of raw rows — subclasses implement ``load`` and register with ``DatasetFactory``."""

    def __init__(self, config: DatasetConfig):
        self.config = config

        self._rows: list[Row] | None = None

    @abc.abstractmethod
    def load(self) -> list[Row]:
        raise NotImplementedError

    def prepare(self) -> "AbstractDataset":
        if self._rows is None:
            self._rows = self.load()

        return self

    @property
    def rows(self) -> list[Row]:
        assert self._rows is not None
        return self._rows

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[Row]:
        return iter(self.rows)

    def __getitem__(self, index: int) -> Row:
        return self.rows[index]
