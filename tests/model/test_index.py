import pytest

from intex.model.config import ModelLayerLocation, ModelLocation
from intex.model.index import IndexElement


def _element(activation: ModelLayerLocation, **index) -> IndexElement:
    names = [ModelLocation(name) for name in index] + [ModelLocation.ACTIVATION]
    return IndexElement(None, names, list(index.values()) + [activation])


def test_remaining_axes_full_slices():
    element = _element(ModelLayerLocation.RESIDUAL_POST, layer=0)
    assert element.remaining_axes() == [ModelLocation.POSITION, ModelLocation.NEURON]


def test_remaining_axes_int_drops_axis():
    element = _element(ModelLayerLocation.RESIDUAL_POST, layer=0, position=3)
    assert element.remaining_axes() == [ModelLocation.NEURON]


def test_remaining_axes_list_keeps_axis():
    element = _element(ModelLayerLocation.ATTENTION_Z, layer=0, head=[0, 1])
    assert element.remaining_axes() == [ModelLocation.POSITION, ModelLocation.HEAD, ModelLocation.NEURON]


def test_remaining_axes_head_activation():
    element = _element(ModelLayerLocation.ATTENTION_Z, layer=0, head=2)
    assert element.remaining_axes() == [ModelLocation.POSITION, ModelLocation.NEURON]


def test_remaining_axes_pattern_activation():
    element = _element(ModelLayerLocation.ATTENTION_PATTERN, layer=0, head=1)
    assert element.remaining_axes() == ["dest_pos", "src_pos"]


def test_remaining_axes_multiple_lists_raise():
    element = _element(ModelLayerLocation.ATTENTION_Z, layer=0, position=[0, 1], head=[0, 1])
    with pytest.raises(ValueError, match="more than one axis"):
        element.remaining_axes()
