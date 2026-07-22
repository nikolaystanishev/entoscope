from types import SimpleNamespace

import pytest
import torch

from intex.lens.config import LensConfig, LensType
from intex.lens.method.factory import LensMethodFactory
from intex.lens.method.lens import AbstractLensMethod
from intex.lens.method.logit_lens import AttentionLogitLens, AttentionQKLogitLens, LogitLens, MLPLogitLens
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


@pytest.mark.parametrize("activation", [ModelLayerLocation.ATTENTION_Z, ModelLayerLocation.ATTENTION_V])
def test_factory_get_returns_attention_logit_lens_for_head_output_activations(activation):
    assert LensMethodFactory.get(LensType.LOGIT_LENS, activation) is AttentionLogitLens


@pytest.mark.parametrize("activation", [ModelLayerLocation.ATTENTION_Q, ModelLayerLocation.ATTENTION_K])
def test_factory_get_returns_qk_logit_lens_for_query_key_activations(activation):
    assert LensMethodFactory.get(LensType.LOGIT_LENS, activation) is AttentionQKLogitLens


def test_factory_get_unprojectable_activation_raises():
    with pytest.raises(ValueError, match="only residual-stream, MLP-hidden and attention head"):
        LensMethodFactory.get(LensType.LOGIT_LENS, ModelLayerLocation.ATTENTION_PATTERN)


def test_factory_get_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown lens type"):
        LensMethodFactory.get("tuned_lens", ModelLayerLocation.RESIDUAL_POST)


@pytest.mark.parametrize(
    ("activation", "expected_class"),
    [
        (ModelLayerLocation.RESIDUAL_POST, LogitLens),
        (ModelLayerLocation.MLP_POST, MLPLogitLens),
        (ModelLayerLocation.ATTENTION_Z, AttentionLogitLens),
        (ModelLayerLocation.ATTENTION_K, AttentionQKLogitLens),
    ],
)
def test_factory_create_builds_configured_instance(activation, expected_class):
    model = object()
    config = _config()

    method = LensMethodFactory.create(config, model, activation)

    assert isinstance(method, expected_class)
    assert method.model is model
    assert method.config is config


# The rejections below are decided from the index and the model config alone, so they never reach
# the weights and a stub model is enough.


def _stub_model(n_heads: int = 8, n_key_value_heads: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(cfg=SimpleNamespace(n_heads=n_heads, n_key_value_heads=n_key_value_heads))


@pytest.mark.parametrize(
    ("method_class", "activation"),
    [(MLPLogitLens, ModelLayerLocation.MLP_POST), (AttentionLogitLens, ModelLayerLocation.ATTENTION_Z)],
)
def test_project_without_a_single_layer_raises(method_class, activation):
    method = method_class(_stub_model(), _config())

    with pytest.raises(ValueError, match="needs a single layer"):
        method.project(torch.zeros(1, 8), _element(activation, layer=[0, 1], position=-1))


@pytest.mark.parametrize(
    ("method_class", "activation"),
    [(AttentionLogitLens, ModelLayerLocation.ATTENTION_V), (AttentionQKLogitLens, ModelLayerLocation.ATTENTION_K)],
)
def test_project_key_value_activation_on_grouped_query_attention_raises(method_class, activation):
    method = method_class(_stub_model(n_heads=8, n_key_value_heads=2), _config())

    with pytest.raises(ValueError, match="grouped-query attention"):
        method.project(torch.zeros(1, 8), _element(activation, layer=0, head=0, position=-1))


@pytest.mark.parametrize("n_key_value_heads", [None, 8])
def test_project_key_value_activation_passes_the_head_check_without_grouping(n_key_value_heads):
    method = AttentionLogitLens(_stub_model(n_key_value_heads=n_key_value_heads), _config())
    element = _element(ModelLayerLocation.ATTENTION_V, layer=0, head=0, position=-1)

    # Past the check it reaches the model's weights, which the stub does not have.
    with pytest.raises(AttributeError):
        method.project(torch.zeros(1, 8), element)
