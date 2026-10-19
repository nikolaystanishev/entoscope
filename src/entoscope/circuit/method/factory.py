from entoscope.circuit.config import CircuitConfig
from entoscope.circuit.method.circuit import AbstractCircuitMethod
from entoscope.circuit.method.information_flow_routes import InformationFlowRoutes
from entoscope.model import Model
from entoscope.operation.registry import RegistryFactory


class CircuitMethodFactory(RegistryFactory[AbstractCircuitMethod]):
    """Resolves the circuit discovery method a config's ``method.type`` names and builds it for a run.

    It holds ``information_flow_routes`` and every method registered with ``register``.
    """

    kind = "circuit method"

    @classmethod
    def create(cls, model: Model, config: CircuitConfig) -> AbstractCircuitMethod:
        """Build the method a config names, once per run. The method then builds one prompt's circuit at a time.

        Args:
            model: The model the method reads.
            config: The run's config, whose ``method.type`` names the method.

        Returns:
            The method.

        Raises:
            ValueError: If no method is registered under the name, or the method rejects the config.
        """
        return cls.get(config.method.type)(model, config)


CircuitMethodFactory.register("information_flow_routes")(InformationFlowRoutes)
