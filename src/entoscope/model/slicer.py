import abc
from collections import defaultdict

from entoscope.model.config import ModelAxis, ModelLayerLocation


class ActivationSlicer(abc.ABC):
    """Orders an index row's values into a slice of one kind of activation tensor.

    Each kind of activation holds its axes in its own order, and a slicer owns that order.

    Attributes:
        AXES: The tensor's axes, in order.
    """

    AXES: tuple[ModelAxis | str, ...]

    @staticmethod
    @abc.abstractmethod
    def extract(index: defaultdict[str, slice | int | list[int]]) -> tuple[slice | int | list[int], ...]:
        """Order a row's values into a slice of the tensor.

        Args:
            index: The row's value on each axis, keyed by axis, a missing axis read whole.

        Returns:
            One index per axis, in ``AXES`` order.
        """
        raise NotImplementedError

    @staticmethod
    def get(activation: ModelLayerLocation) -> type["ActivationSlicer"]:
        """Pick the slicer of an activation.

        Args:
            activation: The activation.

        Returns:
            The slicer for the activation's tensor layout.

        Raises:
            ValueError: If the activation is unknown.
        """
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
    """Slices a ``[batch, pos, neuron]`` activation, such as the residual stream or an MLP's."""

    AXES = ("batch", ModelAxis.POSITION, ModelAxis.NEURON)

    @staticmethod
    def extract(index: defaultdict[str, slice | int | list[int]]) -> tuple[slice | int | list[int], ...]:
        return index["batch"], index[ModelAxis.POSITION], index[ModelAxis.NEURON]


class HeadActivationSlicer(ActivationSlicer):
    """Slices a ``[batch, pos, head, neuron]`` activation: an attention head's query, key, value or output."""

    AXES = ("batch", ModelAxis.POSITION, ModelAxis.HEAD, ModelAxis.NEURON)

    @staticmethod
    def extract(index: defaultdict[str, slice | int | list[int]]) -> tuple[slice | int | list[int], ...]:
        return index["batch"], index[ModelAxis.POSITION], index[ModelAxis.HEAD], index[ModelAxis.NEURON]


class PatternActivationSlicer(ActivationSlicer):
    """Slices a ``[batch, head, dest_pos, src_pos]`` attention pattern, whose positions are read whole."""

    AXES = ("batch", ModelAxis.HEAD, "dest_pos", "src_pos")

    @staticmethod
    def extract(index: defaultdict[str, slice | int | list[int]]) -> tuple[slice | int | list[int], ...]:
        return index["batch"], index[ModelAxis.HEAD], index["dest_pos"], index["src_pos"]
