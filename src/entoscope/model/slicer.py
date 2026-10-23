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
                | ModelLayerLocation.ATTENTION_IN
                | ModelLayerLocation.ATTENTION_OUT_PRE_NORM
                | ModelLayerLocation.ATTENTION_OUT
                | ModelLayerLocation.MLP_IN
                | ModelLayerLocation.MLP_PRE
                | ModelLayerLocation.MLP_PRE_LINEAR
                | ModelLayerLocation.MLP_POST
                | ModelLayerLocation.MLP_OUT_PRE_NORM
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
    """Slices a ``[batch, pos, neuron]`` activation, such as the residual stream, a sublayer's input or an MLP's."""

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
    """Slices a ``[batch, head, dest_pos, src_pos]`` attention pattern: the weight each query gives each key."""

    AXES = ("batch", ModelAxis.HEAD, ModelAxis.DESTINATION_POSITION, ModelAxis.SOURCE_POSITION)

    @staticmethod
    def extract(index: defaultdict[str, slice | int | list[int]]) -> tuple[slice | int | list[int], ...]:
        return (
            index["batch"],
            index[ModelAxis.HEAD],
            index[ModelAxis.DESTINATION_POSITION],
            index[ModelAxis.SOURCE_POSITION],
        )


def located_axes(activation: ModelLayerLocation) -> set[ModelAxis]:
    """Name the axes a locator may name for an activation.

    Those are the axes of the activation's tensor, batch aside, and the layer for an activation inside a
    block, which every activation but ``embed`` is.

    Args:
        activation: The activation.

    Returns:
        The axes.
    """
    axes = {axis for axis in ActivationSlicer.get(activation).AXES if isinstance(axis, ModelAxis)}
    return axes if activation is ModelLayerLocation.EMBEDDING else axes | {ModelAxis.LAYER}
