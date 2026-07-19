from transformer_lens import HookedTransformer

from intex.lens.config import LensConfig, LensType
from intex.lens.method.lens import AbstractLensMethod
from intex.lens.method.logit_lens import LogitLens


class LensMethodFactory:
    @staticmethod
    def get(lens_type: LensType) -> type[AbstractLensMethod]:
        match lens_type:
            case LensType.LOGIT_LENS:
                return LogitLens
            case _:
                raise ValueError(f"Unknown lens type: {lens_type}")

    @classmethod
    def create(cls, config: LensConfig, model: HookedTransformer) -> AbstractLensMethod:
        return cls.get(config.type)(model, config)
