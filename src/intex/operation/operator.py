import abc

from intex.operation.input import OperationInput


class Operator(abc.ABC):
    @abc.abstractmethod
    def apply(self, operation_input: OperationInput, save_dir: str):
        raise NotImplementedError()
