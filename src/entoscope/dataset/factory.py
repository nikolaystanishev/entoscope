from entoscope.dataset.config import DatasetConfig
from entoscope.dataset.dataset import AbstractDataset
from entoscope.operation.registry import RegistryFactory


class DatasetFactory(RegistryFactory[AbstractDataset]):
    """Resolves a config's dataset type to a registered implementation. entoscope ships none."""

    kind = "dataset type"

    @classmethod
    def create(cls, config: DatasetConfig) -> AbstractDataset:
        """Build the dataset a config names, unloaded.

        Args:
            config: The dataset's config, whose ``type`` names the dataset.

        Returns:
            The dataset.

        Raises:
            ValueError: If no dataset is registered under the name.
        """
        return cls.get(config.type)(config)
