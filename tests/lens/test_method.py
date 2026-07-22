from types import SimpleNamespace

import pytest
import torch

from intex.lens.config import LensConfig, LensType
from intex.lens.method import tuned_lens
from intex.lens.method.factory import LensMethodFactory
from intex.lens.method.lens import AbstractLensMethod
from intex.lens.method.logit_lens import AttentionLogitLens, AttentionQKLogitLens, LogitLens, MLPLogitLens
from intex.lens.method.tuned_lens import AttentionTunedLens, MLPTunedLens, TunedLens
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
        LensMethodFactory.get("attention_lens", ModelLayerLocation.RESIDUAL_POST)


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


N_LAYERS = 4
D_MODEL = 3


def _tuned_config(translators: str | None = None) -> LensConfig:
    return LensConfig(
        type=LensType.TUNED_LENS,
        translators=translators,
        source=ModelLocatorConfig(
            activation=ModelLayerLocation.RESIDUAL_POST,
            location=[ModelLocation.LAYER, ModelLocation.POSITION],
        ),
    )


def _translators(bias: bool = True) -> torch.nn.ModuleList:
    """Translator ``i`` scales the residual by ``i`` and shifts it by ``i``, so both are readable."""
    translators = torch.nn.ModuleList()
    for i in range(N_LAYERS):
        linear = torch.nn.Linear(D_MODEL, D_MODEL, bias=bias)
        linear.weight.data = i * torch.eye(D_MODEL)
        if bias:
            linear.bias.data = i * torch.ones(D_MODEL)
        translators.append(linear)
    return translators.requires_grad_(False)


class _TunedModel:
    """A stub the translator cache can key on — unlike ``SimpleNamespace``, it hashes by identity."""

    def __init__(self):
        self.cfg = SimpleNamespace(n_layers=N_LAYERS, d_model=D_MODEL, tokenizer_name="stub")


@pytest.fixture
def tuned_method(monkeypatch):
    def build(bias: bool = True) -> TunedLens:
        monkeypatch.setattr(tuned_lens, "load_translators", lambda *_: _translators(bias))
        return TunedLens(_TunedModel(), _tuned_config())

    return build


@pytest.mark.parametrize(
    ("activation", "expected_class"),
    [
        (ModelLayerLocation.EMBEDDING, TunedLens),
        (ModelLayerLocation.RESIDUAL_PRE, TunedLens),
        (ModelLayerLocation.ATTENTION_OUT, TunedLens),
        (ModelLayerLocation.MLP_OUT, TunedLens),
        (ModelLayerLocation.MLP_POST, MLPTunedLens),
        (ModelLayerLocation.ATTENTION_Z, AttentionTunedLens),
        (ModelLayerLocation.ATTENTION_V, AttentionTunedLens),
    ],
)
def test_factory_get_returns_the_tuned_lens_of_the_activation_space(activation, expected_class):
    assert LensMethodFactory.get(LensType.TUNED_LENS, activation) is expected_class


@pytest.mark.parametrize("activation", [ModelLayerLocation.ATTENTION_Q, ModelLayerLocation.ATTENTION_K])
def test_factory_get_query_key_activation_for_a_tuned_lens_raises(activation):
    with pytest.raises(ValueError, match="Tuned lens cannot project.*use the logit lens"):
        LensMethodFactory.get(LensType.TUNED_LENS, activation)


def test_factory_get_unprojectable_activation_for_a_tuned_lens_raises():
    with pytest.raises(ValueError, match="Tuned lens cannot project.*map onto the vocabulary"):
        LensMethodFactory.get(LensType.TUNED_LENS, ModelLayerLocation.ATTENTION_PATTERN)


@pytest.mark.parametrize(
    ("activation", "layer", "expected_index"),
    [
        (ModelLayerLocation.EMBEDDING, None, 0),
        (ModelLayerLocation.RESIDUAL_PRE, 0, 0),
        (ModelLayerLocation.RESIDUAL_PRE, N_LAYERS - 1, N_LAYERS - 1),
        (ModelLayerLocation.RESIDUAL_MID, 1, 2),
        (ModelLayerLocation.RESIDUAL_POST, 0, 1),
        (ModelLayerLocation.ATTENTION_OUT, 1, 2),
        (ModelLayerLocation.MLP_OUT, 1, 2),
        # Everything the last layer writes lands in the state the model unembeds itself.
        (ModelLayerLocation.RESIDUAL_POST, N_LAYERS - 1, None),
        (ModelLayerLocation.MLP_OUT, N_LAYERS - 1, None),
    ],
)
def test_translator_index_follows_the_residual_state(tuned_method, activation, layer, expected_index):
    index = _element(activation, position=-1) if layer is None else _element(activation, layer=layer, position=-1)

    assert tuned_method()._translator_index(index) == expected_index


@pytest.mark.parametrize("activation", [ModelLayerLocation.RESIDUAL_PRE, ModelLayerLocation.RESIDUAL_MID])
def test_residual_states_are_translated_with_the_bias(tuned_method, activation):
    method = tuned_method()
    residual = torch.ones(1, D_MODEL)

    # ``resid.pre`` of layer 2 and ``resid.mid`` of layer 1 are both translated by translator 2.
    layer = 2 if activation is ModelLayerLocation.RESIDUAL_PRE else 1
    translated = method._to_final_residual(residual, _element(activation, layer=layer, position=-1))

    assert torch.equal(translated, residual + 2 * residual + 2 * torch.ones(D_MODEL))


@pytest.mark.parametrize(
    "activation",
    [ModelLayerLocation.EMBEDDING, ModelLayerLocation.ATTENTION_OUT, ModelLayerLocation.MLP_OUT],
)
def test_component_writes_are_translated_without_the_bias(tuned_method, activation):
    method = tuned_method()
    residual = torch.ones(1, D_MODEL)
    index = 0 if activation is ModelLayerLocation.EMBEDDING else 2

    element = (
        _element(activation, position=-1)
        if activation is ModelLayerLocation.EMBEDDING
        else _element(activation, layer=index - 1, position=-1)
    )
    translated = method._to_final_residual(residual, element)

    assert torch.equal(translated, residual + index * residual)


def test_the_last_residual_state_is_left_untranslated(tuned_method):
    method = tuned_method()
    residual = torch.ones(1, D_MODEL)

    element = _element(ModelLayerLocation.RESIDUAL_POST, layer=N_LAYERS - 1, position=-1)

    assert torch.equal(method._to_final_residual(residual, element), residual)


def test_translators_without_a_bias_are_applied_linearly(tuned_method):
    method = tuned_method(bias=False)
    residual = torch.ones(1, D_MODEL)

    element = _element(ModelLayerLocation.RESIDUAL_POST, layer=1, position=-1)

    assert torch.equal(method._to_final_residual(residual, element), residual + 2 * residual)


def test_translating_without_a_single_layer_raises(tuned_method):
    method = tuned_method()

    with pytest.raises(ValueError, match="needs a single layer"):
        method._to_final_residual(torch.ones(1, D_MODEL), _element(ModelLayerLocation.RESIDUAL_POST, layer=[0, 1]))


def test_resource_id_falls_back_to_the_model_name():
    model = SimpleNamespace(cfg=SimpleNamespace(tokenizer_name=None, model_name="gpt2"))

    assert tuned_lens.resource_id(model) == "gpt2"


def test_resource_id_without_any_model_name_raises():
    model = SimpleNamespace(cfg=SimpleNamespace(tokenizer_name=None, model_name=None))

    with pytest.raises(ValueError, match="Name the translators explicitly"):
        tuned_lens.resource_id(model)


@pytest.mark.parametrize(("d_model", "n_layers"), [(D_MODEL + 1, N_LAYERS), (D_MODEL, N_LAYERS + 1)])
def test_translators_of_another_model_raise(monkeypatch, d_model, n_layers):
    pretrained = SimpleNamespace(
        config=SimpleNamespace(d_model=d_model, num_hidden_layers=n_layers),
        layer_translators=_translators(),
    )
    monkeypatch.setattr(tuned_lens, "Unembed", lambda model: model)
    monkeypatch.setattr(
        tuned_lens.PretrainedTunedLens, "from_unembed_and_pretrained", staticmethod(lambda *_: pretrained)
    )
    tuned_lens.load_translators.cache_clear()

    with pytest.raises(ValueError, match="was trained for a model with"):
        tuned_lens.load_translators(_TunedModel(), "other-model")
