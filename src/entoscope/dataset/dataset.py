import abc
from collections.abc import Iterator
from typing import Any, Generic, cast

from typing_extensions import TypeVar

from entoscope.dataset.config import DatasetConfig

Row = dict[str, Any]

DatasetConfigT = TypeVar("DatasetConfigT", bound=DatasetConfig, default=DatasetConfig)


class AbstractDataset(abc.ABC, Generic[DatasetConfigT]):
    """A collection of raw rows - subclasses implement ``load`` and register with ``DatasetFactory``.

    Its type argument names the dataclass of its config, extending ``DatasetConfig``, which a config file's
    entry is read into; a dataset naming none takes ``DatasetConfig`` itself, which no setting fits.
    """

    def __init__(self, config: DatasetConfig):
        self.config = cast(DatasetConfigT, config)

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
