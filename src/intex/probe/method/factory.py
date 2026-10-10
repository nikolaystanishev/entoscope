from intex.model import Model
from intex.operation.registry import RegistryFactory
from intex.probe.config import ProbeConfig
from intex.probe.method.linear import LinearProbe
from intex.probe.method.probe import AbstractProbeMethod


class ProbeMethodFactory(RegistryFactory[AbstractProbeMethod]):
    """Builds one probe per location - ``create`` is called for every index element."""

    kind = "probe method"

    @classmethod
    def create(cls, config: ProbeConfig, model: Model) -> AbstractProbeMethod:
        return cls.get(config.method.type)(model, config)


ProbeMethodFactory.register("linear")(LinearProbe)
