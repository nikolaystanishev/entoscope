from intex.model import Model
from intex.operation.registry import RegistryFactory
from intex.patching.config import PatchingConfig
from intex.patching.method.attribution import AttributionPatchingMethod, IntegratedGradientsMethod
from intex.patching.method.exact import ExactPatchingMethod
from intex.patching.method.patching import AbstractPatchingMethod


class PatchingMethodFactory(RegistryFactory[AbstractPatchingMethod]):
    """The methods a config's ``method`` names, built in or registered with ``register``."""

    kind = "patching method"

    @classmethod
    def create(cls, config: PatchingConfig, model: Model, batch_size: int = 1) -> AbstractPatchingMethod:
        return cls.get(config.method)(model, config, batch_size)


PatchingMethodFactory.register("exact")(ExactPatchingMethod)
PatchingMethodFactory.register("attribution")(AttributionPatchingMethod)
PatchingMethodFactory.register("integrated_gradients")(IntegratedGradientsMethod)
