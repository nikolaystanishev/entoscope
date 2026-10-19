from entoscope.model import Model
from entoscope.operation.registry import RegistryFactory
from entoscope.patching.config import PatchingConfig
from entoscope.patching.method.attribution import AttributionPatchingMethod, IntegratedGradientsMethod
from entoscope.patching.method.exact import ExactPatchingMethod
from entoscope.patching.method.patching import AbstractPatchingMethod


class PatchingMethodFactory(RegistryFactory[AbstractPatchingMethod]):
    """Resolves the patching method a config's ``method.type`` names and builds it for a run.

    It holds ``exact``, ``attribution``, ``integrated_gradients`` and every method registered with
    ``register``.
    """

    kind = "patching method"

    @classmethod
    def create(cls, model: Model, config: PatchingConfig) -> AbstractPatchingMethod:
        """Build the method a config names, once per run.

        Args:
            model: The model the method patches.
            config: The run's config, whose ``method.type`` names the method.

        Returns:
            The method.

        Raises:
            ValueError: If no method is registered under the name, or the method rejects the config.
        """
        return cls.get(config.method.type)(model, config)


PatchingMethodFactory.register("exact")(ExactPatchingMethod)
PatchingMethodFactory.register("attribution")(AttributionPatchingMethod)
PatchingMethodFactory.register("integrated_gradients")(IntegratedGradientsMethod)
