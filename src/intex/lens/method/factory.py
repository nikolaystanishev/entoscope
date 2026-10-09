from intex.lens.config import LensConfig
from intex.lens.method.lens import AbstractLensMethod
from intex.lens.method.logit_lens import LogitLens
from intex.lens.method.parameter_lens import ParameterLens
from intex.lens.method.tuned_lens import TunedLens
from intex.model import Model
from intex.operation.registry import RegistryFactory


class LensMethodFactory(RegistryFactory[AbstractLensMethod]):
    """The lenses a config's ``method`` names, built in or registered with ``register``."""

    kind = "lens method"

    @classmethod
    def create(cls, config: LensConfig, model: Model) -> AbstractLensMethod:
        return cls.get(config.method)(model, config)


LensMethodFactory.register("logit_lens")(LogitLens)
LensMethodFactory.register("tuned_lens")(TunedLens)
LensMethodFactory.register("parameter_lens")(ParameterLens)
