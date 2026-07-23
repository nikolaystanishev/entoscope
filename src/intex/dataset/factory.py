from collections.abc import Callable

from intex.dataset.config import DatasetConfig
from intex.dataset.dataset import AbstractDataset


class DatasetFactory:
    """Resolves a config's dataset type to a registered implementation."""

    _registry: dict[str, type[AbstractDataset]] = {}

    @classmethod
    def register(cls, dataset_type: str) -> Callable[[type[AbstractDataset]], type[AbstractDataset]]:
        def decorator(dataset_class: type[AbstractDataset]) -> type[AbstractDataset]:
            registered = cls._registry.get(dataset_type)
            if registered is not None and registered is not dataset_class:
                raise ValueError(
                    f"Dataset type '{dataset_type}' is already registered to {registered.__name__}; "
                    f"{dataset_class.__name__} must use another name"
                )
            cls._registry[dataset_type] = dataset_class
            return dataset_class

        return decorator

    @classmethod
    def get(cls, dataset_type: str) -> type[AbstractDataset]:
        if dataset_type not in cls._registry:
            raise ValueError(
                f"Unknown dataset type: {dataset_type}. Registered types: {cls.registered()}. "
                "A dataset is registered when its module is imported"
            )
        return cls._registry[dataset_type]

    @classmethod
    def registered(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def create(cls, config: DatasetConfig) -> AbstractDataset:
        return cls.get(config.type)(config)
