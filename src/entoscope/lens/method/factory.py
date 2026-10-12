from entoscope.lens.config import LensConfig
from entoscope.lens.method.lens import AbstractLensMethod
from entoscope.lens.method.logit_lens import LogitLens
from entoscope.lens.method.parameter_lens import ParameterLens
from entoscope.lens.method.tuned_lens import TunedLens
from entoscope.model import Model
from entoscope.operation.registry import RegistryFactory


class LensMethodFactory(RegistryFactory[AbstractLensMethod]):
    """The lenses a config's ``method`` names, built in or registered with ``register``."""

    kind = "lens method"

    @classmethod
    def create(cls, config: LensConfig, model: Model) -> AbstractLensMethod:
        return cls.get(config.method.type)(model, config)


LensMethodFactory.register("logit_lens")(LogitLens)
LensMethodFactory.register("tuned_lens")(TunedLens)
LensMethodFactory.register("parameter_lens")(ParameterLens)
