from intex.lens.config import LensConfig, LensType
from intex.lens.method.lens import AbstractLensMethod
from intex.lens.method.logit_lens import AttentionLogitLens, AttentionQKLogitLens, LogitLens, MLPLogitLens
from intex.lens.method.parameter_lens import (
    AbstractParameterLensMethod,
    AttentionReadParameterLens,
    AttentionWriteParameterLens,
    MLPKeyParameterLens,
    MLPValueParameterLens,
)
from intex.lens.method.tuned_lens import AttentionTunedLens, MLPTunedLens, TunedLens
from intex.model import Model
from intex.model.config import ModelLayerLocation


class LensMethodFactory:
    """Builds one lens per index element — the class follows the lens type and the location it names."""

    @classmethod
    def get(cls, lens_type: LensType, activation: ModelLayerLocation) -> type[AbstractLensMethod]:
        match lens_type:
            case LensType.LOGIT_LENS:
                return cls._logit_lens(activation)
            case LensType.TUNED_LENS:
                return cls._tuned_lens(activation)
            case LensType.PARAMETER_LENS:
                return cls._parameter_lens(activation)
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

    @staticmethod
    def _tuned_lens(activation: ModelLayerLocation) -> type[AbstractLensMethod]:
        match activation:
            case (
                ModelLayerLocation.EMBEDDING
                | ModelLayerLocation.RESIDUAL_PRE
                | ModelLayerLocation.RESIDUAL_MID
                | ModelLayerLocation.RESIDUAL_POST
                | ModelLayerLocation.ATTENTION_OUT
                | ModelLayerLocation.MLP_OUT
            ):
                return TunedLens
            case ModelLayerLocation.MLP_PRE | ModelLayerLocation.MLP_PRE_LINEAR | ModelLayerLocation.MLP_POST:
                return MLPTunedLens
            case ModelLayerLocation.ATTENTION_V | ModelLayerLocation.ATTENTION_Z:
                return AttentionTunedLens
            case ModelLayerLocation.ATTENTION_Q | ModelLayerLocation.ATTENTION_K:
                raise ValueError(
                    f"Tuned lens cannot project {activation}: its translator maps the residual stream onto the "
                    "final residual, while queries and keys are read back through the other side of the head — "
                    "use the logit lens for them"
                )
            case _:
                raise ValueError(
                    f"Tuned lens cannot project {activation}: only residual-stream, MLP-hidden and attention head "
                    "activations map onto the vocabulary"
                )

    @staticmethod
    def _parameter_lens(activation: ModelLayerLocation) -> type[AbstractParameterLensMethod]:
        match activation:
            case ModelLayerLocation.MLP_PRE | ModelLayerLocation.MLP_PRE_LINEAR:
                return MLPKeyParameterLens
            case ModelLayerLocation.MLP_POST:
                return MLPValueParameterLens
            case ModelLayerLocation.ATTENTION_Q | ModelLayerLocation.ATTENTION_K | ModelLayerLocation.ATTENTION_V:
                return AttentionReadParameterLens
            case ModelLayerLocation.ATTENTION_Z:
                return AttentionWriteParameterLens
            case _:
                raise ValueError(
                    f"Parameter lens cannot project {activation}: only MLP-hidden and attention head locations name "
                    "a weight vector, one per neuron or head dimension. Residual-stream locations ('embed', "
                    "'resid.*', 'attn.out', 'mlp.out') index the residual basis instead of a weight"
                )

    @classmethod
    def create(cls, config: LensConfig, model: Model, activation: ModelLayerLocation) -> AbstractLensMethod:
        return cls.get(config.type, activation)(model, config)
