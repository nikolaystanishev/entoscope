from intex.circuit.config import CircuitConfig, CircuitMethodType
from intex.circuit.method.circuit import AbstractCircuitMethod
from intex.circuit.method.information_flow_routes import InformationFlowRoutes
from intex.model import Model


class CircuitMethodFactory:
    @staticmethod
    def get(method_type: CircuitMethodType) -> type[AbstractCircuitMethod]:
        match method_type:
            case CircuitMethodType.INFORMATION_FLOW_ROUTES:
                return InformationFlowRoutes
            case _:
                raise ValueError(f"Unknown circuit method type: {method_type}")

    @classmethod
    def create(cls, config: CircuitConfig, model: Model) -> AbstractCircuitMethod:
        return cls.get(config.type)(model, config)
