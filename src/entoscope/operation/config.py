import dataclasses
import os
from typing import Generic, TypeVar, get_args

from entoscope.util.data import from_dict, read_yaml, to_dict, write_json

CONFIG_FILE = "config.json"


@dataclasses.dataclass(kw_only=True)
class AbstractConfig:
    """What every operation's config shares.

    ``config_id`` names the YAML entry the config was read from - ``ConfigFactory`` sets it - and
    titles the run's plan and result pages; a config built in code may leave it unset.
    """

    config_id: str | None = None

    def save(self, save_dir: str) -> None:
        """Write the config a run was made under beside that run's results."""
        write_json(os.path.join(save_dir, CONFIG_FILE), to_dict(self))


T = TypeVar("T", bound=AbstractConfig)


class ConfigFactory(Generic[T]):
    """Reads configs out of a YAML file of ``config_id: config`` entries.

    The type argument is the class an entry is read into, so it is required:
    ``ConfigFactory[LensConfig]("configs.yaml").create("my_lens")``. Reading is strict - an unknown
    or missing key raises - and the file is read once per factory. A value may name environment
    variables as ``${NAME}``: each is replaced by the variable's value, or by ``NAME`` itself when it
    is unset, and a value that is a single variable holding digits becomes an int.
    """

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
