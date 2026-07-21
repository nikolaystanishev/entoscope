from intex.model import Model
from intex.probe.config import ProbeConfig, ProbeType
from intex.probe.method.linear import LinearProbe
from intex.probe.method.probe import AbstractProbeMethod


class ProbeMethodFactory:
    """Builds one probe per location — ``create`` is called for every index element."""

    @staticmethod
    def get(probe_type: ProbeType) -> type[AbstractProbeMethod]:
        match probe_type:
            case ProbeType.LINEAR:
                return LinearProbe
            case _:
                raise ValueError(f"Unknown probe type: {probe_type}")

    @classmethod
    def create(cls, config: ProbeConfig, model: Model) -> AbstractProbeMethod:
        return cls.get(config.type)(model, config)
