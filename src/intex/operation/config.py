import abc
from typing import Generic, TypeVar, get_args

from transformer_lens import HookedTransformer

from intex.operation.input import OperationInput
from intex.util.data import read_yaml


class AbstractConfig(abc.ABC):
    @abc.abstractmethod
    def prepare(self, model: HookedTransformer, operation_input: OperationInput):
        raise NotImplementedError


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
        if config_id not in self.config:
            raise KeyError(f"Config with id {config_id} does not exist in the provided config file: {self.config_path}")
        return get_args(getattr(self, "__orig_class__", None))[0](config_id, self._get_config()[config_id])
