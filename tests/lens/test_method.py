import pytest
import torch

from intex.lens.config import LensConfig, LensType
from intex.lens.method.factory import LensMethodFactory
from intex.lens.method.lens import AbstractLensMethod
from intex.lens.method.logit_lens import LogitLens, MLPLogitLens
from intex.model.config import ModelLayerLocation, ModelLocation, ModelLocatorConfig
from intex.model.index import IndexElement

RESIDUAL_ACTIVATIONS = [
    ModelLayerLocation.EMBEDDING,
    ModelLayerLocation.RESIDUAL_PRE,
    ModelLayerLocation.RESIDUAL_MID,
    ModelLayerLocation.RESIDUAL_POST,
    ModelLayerLocation.ATTENTION_OUT,
    ModelLayerLocation.MLP_OUT,
]

MLP_HIDDEN_ACTIVATIONS = [
    ModelLayerLocation.MLP_PRE,
    ModelLayerLocation.MLP_PRE_LINEAR,
    ModelLayerLocation.MLP_POST,
]


def _config() -> LensConfig:
    return LensConfig(
        type=LensType.LOGIT_LENS,
        source=ModelLocatorConfig(
            activation=ModelLayerLocation.RESIDUAL_POST,
            location=[ModelLocation.LAYER, ModelLocation.POSITION],
        ),
    )


def _element(activation: ModelLayerLocation, **index) -> IndexElement:
    names = [ModelLocation(name) for name in index] + [ModelLocation.ACTIVATION]
    return IndexElement(None, names, list(index.values()) + [activation])


def test_abstract_method_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AbstractLensMethod(object(), _config())


@pytest.mark.parametrize("activation", RESIDUAL_ACTIVATIONS)
def test_factory_get_returns_logit_lens_for_residual_activations(activation):
    assert LensMethodFactory.get(LensType.LOGIT_LENS, activation) is LogitLens


@pytest.mark.parametrize("activation", MLP_HIDDEN_ACTIVATIONS)
def test_factory_get_returns_mlp_logit_lens_for_mlp_hidden_activations(activation):
    assert LensMethodFactory.get(LensType.LOGIT_LENS, activation) is MLPLogitLens


@pytest.mark.parametrize(
    "activation",
    [ModelLayerLocation.ATTENTION_Q, ModelLayerLocation.ATTENTION_Z, ModelLayerLocation.ATTENTION_PATTERN],
)
def test_factory_get_unprojectable_activation_raises(activation):
    with pytest.raises(ValueError, match="only residual-stream and MLP-hidden"):
        LensMethodFactory.get(LensType.LOGIT_LENS, activation)


def test_factory_get_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown lens type"):
        LensMethodFactory.get("tuned_lens", ModelLayerLocation.RESIDUAL_POST)


@pytest.mark.parametrize(
    ("activation", "expected_class"),
    [(ModelLayerLocation.RESIDUAL_POST, LogitLens), (ModelLayerLocation.MLP_POST, MLPLogitLens)],
)
def test_factory_create_builds_configured_instance(activation, expected_class):
    model = object()
    config = _config()

    method = LensMethodFactory.create(config, model, activation)

    assert isinstance(method, expected_class)
    assert method.model is model
    assert method.config is config


def test_project_mlp_hidden_without_a_single_layer_raises():
    # Decided from the index alone, so the projection never reaches the model.
    method = MLPLogitLens(object(), _config())

    with pytest.raises(ValueError, match="needs a single layer"):
        method.project(torch.zeros(1, 8), _element(ModelLayerLocation.MLP_POST, layer=[0, 1], position=-1))
