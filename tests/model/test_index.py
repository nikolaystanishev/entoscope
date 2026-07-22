import pandas as pd
import pytest

from intex.model.config import ModelLayerLocation, ModelLocation, ModelLocationIndex, ModelLocatorConfig
from intex.model.index import Index, IndexElement


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


def test_get_returns_the_indexed_value():
    element = _element(ModelLayerLocation.MLP_POST, layer=1, neuron=373)

    assert element.get(ModelLocation.NEURON) == 373
    assert element.get(ModelLocation.ACTIVATION) == ModelLayerLocation.MLP_POST


def test_get_missing_location_is_none():
    element = _element(ModelLayerLocation.MLP_POST, layer=1)

    assert element.get(ModelLocation.NEURON) is None


def test_layer_only_for_a_single_layer():
    assert _element(ModelLayerLocation.MLP_POST, layer=2).layer() == 2
    assert _element(ModelLayerLocation.MLP_POST, layer=[0, 1]).layer() is None
    assert _element(ModelLayerLocation.MLP_POST, position=0).layer() is None


def _source_index(df: pd.DataFrame) -> Index:
    # ``source_activation_names`` reads only the source df, so the config is a placeholder.
    return Index(ModelLocatorConfig(), ModelLocationIndex(df))


def test_source_activation_names_dedupes_over_non_layer_axes():
    # A layer x position grid over resid.post: names depend only on the layer, so the two positions
    # collapse to one hook name per layer.
    df = pd.DataFrame(
        {
            ModelLocation.LAYER: [0, 0, 1, 1],
            ModelLocation.POSITION: [0, 1, 0, 1],
            ModelLocation.ACTIVATION: [ModelLayerLocation.RESIDUAL_POST] * 4,
        }
    )
    assert _source_index(df).source_activation_names() == {
        "blocks.0.hook_resid_post",
        "blocks.1.hook_resid_post",
    }


def test_source_activation_names_mixed_activations():
    # Coordinate-style rows carrying their own activation resolve to per-row hook names.
    df = pd.DataFrame(
        {
            ModelLocation.LAYER: [10, 11],
            ModelLocation.ACTIVATION: [ModelLayerLocation.RESIDUAL_POST, ModelLayerLocation.MLP_OUT],
        }
    )
    assert _source_index(df).source_activation_names() == {
        "blocks.10.hook_resid_post",
        "blocks.11.hook_mlp_out",
    }


def test_source_activation_names_empty_index_is_empty():
    df = pd.DataFrame({ModelLocation.ACTIVATION: pd.Series([], dtype=object)})
    assert _source_index(df).source_activation_names() == set()
