from entoscope.circuit.config import CircuitConfig
from entoscope.circuit.method.circuit import AbstractCircuitMethod
from entoscope.circuit.method.information_flow_routes import InformationFlowRoutes
from entoscope.model import Model
from entoscope.operation.registry import RegistryFactory


class CircuitMethodFactory(RegistryFactory[AbstractCircuitMethod]):
    """Resolves the circuit-discovery method a config's ``method.type`` names and builds it for a run.

    It holds ``information_flow_routes`` and every method registered with ``register``. ``create`` builds the
    method once per run, from the model and the config; the method then builds one prompt's circuit at a time.
    """

    kind = "circuit method"

    @classmethod
    def create(cls, model: Model, config: CircuitConfig) -> AbstractCircuitMethod:
        return cls.get(config.method.type)(model, config)


CircuitMethodFactory.register("information_flow_routes")(InformationFlowRoutes)
