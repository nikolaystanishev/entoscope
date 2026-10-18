from entoscope.lens.config import LensConfig
from entoscope.lens.method.lens import AbstractLensMethod
from entoscope.lens.method.logit_lens import LogitLens
from entoscope.lens.method.parameter_lens import ParameterLens
from entoscope.lens.method.tuned_lens import TunedLens
from entoscope.model import Model
from entoscope.operation.registry import RegistryFactory


class LensMethodFactory(RegistryFactory[AbstractLensMethod]):
    """Resolves the lens a config's ``method.type`` names and builds it for a run.

    It holds ``logit_lens``, ``tuned_lens``, ``parameter_lens`` and every lens registered with
    ``register``.
    """

    kind = "lens method"

    @classmethod
    def create(cls, model: Model, config: LensConfig) -> AbstractLensMethod:
        """Build the lens a config names, once per run.

        Args:
            model: The model the lens reads.
            config: The run's config, whose ``method.type`` names the lens.

        Returns:
            The lens.

        Raises:
            ValueError: If no lens is registered under the name, or the lens rejects the config.
        """
        return cls.get(config.method.type)(model, config)


LensMethodFactory.register("logit_lens")(LogitLens)
LensMethodFactory.register("tuned_lens")(TunedLens)
LensMethodFactory.register("parameter_lens")(ParameterLens)
