import dataclasses
import os
from typing import Generic, TypeVar, get_args

from intex.util.collections import to_dict
from intex.util.data import from_dict, read_yaml, write_json

CONFIG_FILE = "config.json"


@dataclasses.dataclass(kw_only=True)
class AbstractConfig:
    config_id: str | None = None

    def save(self, save_dir: str) -> None:
        """Write the config a run was made under beside that run's results."""
        write_json(os.path.join(save_dir, CONFIG_FILE), to_dict(self))


T = TypeVar("T", bound=AbstractConfig)


class ConfigFactory(Generic[T]):
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = None

    def _get_config(self) -> dict | list:
        if self.config is None:
            self.config = read_yaml(self.config_path)
        return self.config

    def create(self, config_id: str) -> T:
        if config_id not in self._get_config():
            raise KeyError(f"Config with id {config_id} does not exist in the provided config file: {self.config_path}")

        config = from_dict(get_args(getattr(self, "__orig_class__", None))[0], self._get_config()[config_id])
        config.config_id = config_id
        return config
