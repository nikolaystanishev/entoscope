import abc

from intex.operation.input import OperationInput
from intex.operation.plan import OperationPlan


class Operator(abc.ABC):
    @abc.abstractmethod
    def apply(self, operation_input: OperationInput, save_dir: str):
        raise NotImplementedError()

    @abc.abstractmethod
    def plan(self, operation_input: OperationInput, save_dir: str | None = None, verbose: bool = True) -> OperationPlan:
        raise NotImplementedError()
