from entoscope.lens.config import LensConfig
from entoscope.lens.method.lens import AbstractLensMethod
from entoscope.lens.method.logit_lens import LogitLens
from entoscope.lens.method.parameter_lens import ParameterLens
from entoscope.lens.method.tuned_lens import TunedLens
from entoscope.model import Model
from entoscope.operation.registry import RegistryFactory


class LensMethodFactory(RegistryFactory[AbstractLensMethod]):
    """Resolves the lens a config's ``method.type`` names and builds it for a run.

    It holds ``logit_lens``, ``tuned_lens``, ``parameter_lens`` and every lens registered with ``register``.
    ``create`` builds the lens once per run, from the model and the config.
    """

    kind = "lens method"

    @classmethod
    def create(cls, model: Model, config: LensConfig) -> AbstractLensMethod:
        return cls.get(config.method.type)(model, config)


LensMethodFactory.register("logit_lens")(LogitLens)
LensMethodFactory.register("tuned_lens")(TunedLens)
LensMethodFactory.register("parameter_lens")(ParameterLens)
