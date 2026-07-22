from intex.lens.config import LensConfig, LensType
from intex.lens.method.lens import AbstractLensMethod
from intex.lens.method.logit_lens import AttentionLogitLens, AttentionQKLogitLens, LogitLens, MLPLogitLens
from intex.model import Model
from intex.model.config import ModelLayerLocation


class LensMethodFactory:
    """Builds one lens per index element — the class follows the space the activation lives in."""

    @classmethod
    def get(cls, lens_type: LensType, activation: ModelLayerLocation) -> type[AbstractLensMethod]:
        match lens_type:
            case LensType.LOGIT_LENS:
                return cls._logit_lens(activation)
            case _:
                raise ValueError(f"Unknown lens type: {lens_type}")

    @staticmethod
    def _logit_lens(activation: ModelLayerLocation) -> type[AbstractLensMethod]:
        match activation:
            case (
                ModelLayerLocation.EMBEDDING
                | ModelLayerLocation.RESIDUAL_PRE
                | ModelLayerLocation.RESIDUAL_MID
                | ModelLayerLocation.RESIDUAL_POST
                | ModelLayerLocation.ATTENTION_OUT
                | ModelLayerLocation.MLP_OUT
            ):
                return LogitLens
            case ModelLayerLocation.MLP_PRE | ModelLayerLocation.MLP_PRE_LINEAR | ModelLayerLocation.MLP_POST:
                return MLPLogitLens
            case ModelLayerLocation.ATTENTION_V | ModelLayerLocation.ATTENTION_Z:
                return AttentionLogitLens
            case ModelLayerLocation.ATTENTION_Q | ModelLayerLocation.ATTENTION_K:
                return AttentionQKLogitLens
            case _:
                raise ValueError(
                    f"Logit lens cannot project {activation}: only residual-stream, MLP-hidden and attention head "
                    "activations map onto the vocabulary"
                )

    @classmethod
    def create(cls, config: LensConfig, model: Model, activation: ModelLayerLocation) -> AbstractLensMethod:
        return cls.get(config.type, activation)(model, config)
