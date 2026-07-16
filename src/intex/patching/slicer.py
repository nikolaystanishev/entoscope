import abc
from collections import defaultdict

from intex.model.config import ModelLayerLocation, ModelLocation


class ActivationSlicer(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def extract(index: defaultdict[str, slice | int | list[int]]) -> tuple[slice | int | list[int], ...]:
        raise NotImplementedError

    @staticmethod
    def get(activation: ModelLayerLocation) -> type["ActivationSlicer"]:
        match activation:
            case (
                ModelLayerLocation.EMBEDDING
                | ModelLayerLocation.RESIDUAL_PRE
                | ModelLayerLocation.RESIDUAL_MID
                | ModelLayerLocation.RESIDUAL_POST
                | ModelLayerLocation.ATTENTION_OUT
                | ModelLayerLocation.MLP_PRE
                | ModelLayerLocation.MLP_PRE_LINEAR
                | ModelLayerLocation.MLP_POST
                | ModelLayerLocation.MLP_OUT
            ):
                return LinearActivationSlicer
            case (
                ModelLayerLocation.ATTENTION_Q
                | ModelLayerLocation.ATTENTION_K
                | ModelLayerLocation.ATTENTION_V
                | ModelLayerLocation.ATTENTION_Z
            ):
                return HeadActivationSlicer
            case ModelLayerLocation.ATTENTION_PATTERN:
                return PatternActivationSlicer
            case _:
                raise ValueError(f"Unknown activation: {activation}")


class LinearActivationSlicer(ActivationSlicer):
    @staticmethod
    def extract(index: defaultdict[str, slice | int | list[int]]) -> tuple[slice | int | list[int], ...]:
        # [batch, pos, neuron]
        return index["batch"], index[ModelLocation.POSITION], index[ModelLocation.NEURON]


class HeadActivationSlicer(ActivationSlicer):
    @staticmethod
    def extract(index: defaultdict[str, slice | int | list[int]]) -> tuple[slice | int | list[int], ...]:
        # [batch, pos, head, neuron]
        return index["batch"], index[ModelLocation.POSITION], index[ModelLocation.HEAD], index[ModelLocation.NEURON]


class PatternActivationSlicer(ActivationSlicer):
    @staticmethod
    def extract(index: defaultdict[str, slice | int | list[int]]) -> tuple[slice | int | list[int], ...]:
        # [batch, head, dest_pos, src_pos]
        return index["batch"], index[ModelLocation.HEAD], index["dest_pos"], index["src_pos"]
