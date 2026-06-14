import abc

from intex.operation.config import AbstractConfig
from intex.operation.input import OperationInput


class Operator(abc.ABC):
    @abc.abstractmethod
    def apply(self, config: AbstractConfig, operation_input: OperationInput, save_dir: str):
        raise NotImplementedError()
