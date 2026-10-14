from entoscope.model import Model
from entoscope.operation.registry import RegistryFactory
from entoscope.patching.config import PatchingConfig
from entoscope.patching.method.attribution import AttributionPatchingMethod, IntegratedGradientsMethod
from entoscope.patching.method.exact import ExactPatchingMethod
from entoscope.patching.method.patching import AbstractPatchingMethod


class PatchingMethodFactory(RegistryFactory[AbstractPatchingMethod]):
    """The methods a config's ``method`` names, built in or registered with ``register``."""

    kind = "patching method"

    @classmethod
    def create(cls, config: PatchingConfig, model: Model, batch_size: int = 1) -> AbstractPatchingMethod:
        return cls.get(config.method.type)(model, config, batch_size)


PatchingMethodFactory.register("exact")(ExactPatchingMethod)
PatchingMethodFactory.register("attribution")(AttributionPatchingMethod)
PatchingMethodFactory.register("integrated_gradients")(IntegratedGradientsMethod)
