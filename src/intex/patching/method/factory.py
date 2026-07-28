from intex.model import Model
from intex.patching.config import PatchingConfig, PatchingMethodType
from intex.patching.method.attribution import AttributionPatchingMethod, IntegratedGradientsMethod
from intex.patching.method.exact import ExactPatchingMethod
from intex.patching.method.patching import AbstractPatchingMethod


class PatchingMethodFactory:
    @staticmethod
    def get(method_type: PatchingMethodType) -> type[AbstractPatchingMethod]:
        match method_type:
            case PatchingMethodType.EXACT:
                return ExactPatchingMethod
            case PatchingMethodType.ATTRIBUTION:
                return AttributionPatchingMethod
            case PatchingMethodType.INTEGRATED_GRADIENTS:
                return IntegratedGradientsMethod
            case _:
                raise ValueError(f"Unknown patching method: {method_type}")

    @classmethod
    def create(cls, config: PatchingConfig, model: Model, batch_size: int = 1) -> AbstractPatchingMethod:
        return cls.get(config.method)(model, config, batch_size)
