from intex.circuit.config import CircuitConfig
from intex.circuit.method.circuit import AbstractCircuitMethod
from intex.circuit.method.information_flow_routes import InformationFlowRoutes
from intex.model import Model
from intex.operation.registry import RegistryFactory


class CircuitMethodFactory(RegistryFactory[AbstractCircuitMethod]):
    """The methods a config's ``method`` names, built in or registered with ``register``."""

    kind = "circuit method"

    @classmethod
    def create(cls, config: CircuitConfig, model: Model) -> AbstractCircuitMethod:
        return cls.get(config.method.type)(model, config)


CircuitMethodFactory.register("information_flow_routes")(InformationFlowRoutes)
