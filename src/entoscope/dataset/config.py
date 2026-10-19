from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from entoscope.operation.config import AbstractConfig
from entoscope.operation.method import MethodConfig

if TYPE_CHECKING:
    from entoscope.dataset.factory import DatasetFactory


@dataclass(kw_only=True)
class DatasetConfig(AbstractConfig, MethodConfig):
    """The base of every dataset's config, whose ``type`` names the registered dataset.

    A dataset with settings of its own names a dataclass extending this one. A config file's entry is
    read strictly into the class its ``type``'s dataset names, as a method config is, so a key that
    class does not declare fails as the file is read.
    """

    @classmethod
    def factory(cls) -> type[DatasetFactory]:
        """Name the factory datasets register with.

        Returns:
            ``DatasetFactory``.
        """
        from entoscope.dataset.factory import DatasetFactory

        return DatasetFactory
