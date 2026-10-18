from entoscope.model import Model
from entoscope.operation.registry import RegistryFactory
from entoscope.probe.config import ProbeConfig
from entoscope.probe.method.linear import LinearProbe
from entoscope.probe.method.probe import AbstractProbeMethod


class ProbeMethodFactory(RegistryFactory[AbstractProbeMethod]):
    """Resolves the probe a config's ``method.type`` names and builds one per location.

    It holds ``linear`` and every probe registered with ``register``.
    """

    kind = "probe method"

    @classmethod
    def create(cls, model: Model, config: ProbeConfig) -> AbstractProbeMethod:
        """Build a fresh, unfitted probe of the kind a config names, called once per index element.

        Args:
            model: The model the features are read from.
            config: The run's config, whose ``method.type`` names the probe.

        Returns:
            The probe.

        Raises:
            ValueError: If no probe is registered under the name, or the probe rejects the config.
        """
        return cls.get(config.method.type)(model, config)


ProbeMethodFactory.register("linear")(LinearProbe)
