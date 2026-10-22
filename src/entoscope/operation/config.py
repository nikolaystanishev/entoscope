import dataclasses
import os
from typing import Generic, TypeVar, get_args

from entoscope.util.data import from_dict, read_yaml, to_dict, write_json

CONFIG_FILE = "config.json"


@dataclasses.dataclass(kw_only=True)
class AbstractConfig:
    """The base of every operation's config.

    Attributes:
        config_id: The id of the YAML entry the config was read from, set by ``ConfigFactory``. It titles
            the run's plan and result pages, and a config built in code may leave it unset.
    """

    config_id: str | None = None

    def save(self, save_dir: str) -> None:
        """Write the config as ``config.json`` beside a run's results.

        Args:
            save_dir: The directory the run's results are written to.

        Raises:
            TypeError: If a field holds a value with no JSON form, such as a function or a tensor.
        """
        write_json(os.path.join(save_dir, CONFIG_FILE), to_dict(self))


T = TypeVar("T", bound=AbstractConfig)


class ConfigFactory(Generic[T]):
    """Reads configs out of a YAML file of ``config_id: config`` entries.

    The type argument is the class an entry is read into, so it is required:
    ``ConfigFactory[LensConfig]("configs.yaml").create("my_lens")``. Reading is strict, so an unknown
    or missing key raises, and the file is read once per factory. A value may name environment
    variables as ``${NAME}``: each is replaced by the variable's value, or by ``NAME`` itself when it
    is unset, and a value that is a single variable holding digits becomes an int.

    Args:
        config_path: The YAML file to read.
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = None

    def _get_config(self) -> dict | list:
        if self.config is None:
            self.config = read_yaml(self.config_path)
        return self.config

    def create(self, config_id: str) -> T:
        """Read one entry of the file into the factory's config class.

        Args:
            config_id: The entry to read, which also becomes the config's ``config_id``.

        Returns:
            The config the entry describes.

        Raises:
            KeyError: If the file has no entry ``config_id``.
            ValueError: If the entry has a key the config does not, lacks a required one, or holds a value
                its field cannot take.
        """
        if config_id not in self._get_config():
            raise KeyError(f"Config with id {config_id} does not exist in the provided config file: {self.config_path}")

        config = from_dict(get_args(getattr(self, "__orig_class__", None))[0], self._get_config()[config_id])
        config.config_id = config_id
        return config
