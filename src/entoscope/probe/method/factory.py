from entoscope.model import Model
from entoscope.operation.registry import RegistryFactory
from entoscope.probe.config import ProbeConfig
from entoscope.probe.method.linear import LinearProbe
from entoscope.probe.method.probe import AbstractProbeMethod


class ProbeMethodFactory(RegistryFactory[AbstractProbeMethod]):
    """Builds one probe per location - ``create`` is called for every index element."""

    kind = "probe method"

    @classmethod
    def create(cls, model: Model, config: ProbeConfig) -> AbstractProbeMethod:
        return cls.get(config.method.type)(model, config)


ProbeMethodFactory.register("linear")(LinearProbe)
