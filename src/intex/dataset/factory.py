from intex.dataset.config import DatasetConfig
from intex.dataset.dataset import AbstractDataset
from intex.operation.registry import RegistryFactory


class DatasetFactory(RegistryFactory[AbstractDataset]):
    """Resolves a config's dataset type to a registered implementation - intex ships none."""

    kind = "dataset type"

    @classmethod
    def create(cls, config: DatasetConfig) -> AbstractDataset:
        return cls.get(config.type)(config)
