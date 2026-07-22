import json

import pandas as pd
import pytest
import torch
import yaml

from intex.lens.collector import escape_token
from intex.lens.config import LensConfig, LensType, ProjectionSpace
from intex.lens.input import LensInput
from intex.lens.lens import Lens
from intex.lens.method import tuned_lens
from intex.lens.method.factory import LensMethodFactory
from intex.lens.method.lens import AbstractLensMethod
from intex.lens.method.logit_lens import AttentionLogitLens, AttentionQKLogitLens, LogitLens, MLPLogitLens
from intex.lens.method.parameter_lens import (
    AttentionReadParameterLens,
    AttentionWriteParameterLens,
    MLPKeyParameterLens,
    MLPValueParameterLens,
)
from intex.lens.method.tuned_lens import TunedLens
from intex.model.config import Filter, ModelLayerLocation, ModelLocation, ModelLocatorConfig
from intex.model.index import IndexElement
from intex.operation.config import ConfigFactory

PROMPT = "The capital of France is"
LAYER = 8
NEURON = 373
HEAD = 7

LENS_YAML = {
    "my_lens": {
        "type": "logit_lens",
        "source": {"activation": "resid.post", "location": ["layer", "position"]},
        "top_k": 2,
    }
}


def _declarative_config(top_k: int = 1) -> LensConfig:
    return LensConfig(
        type=LensType.LOGIT_LENS,
        source=ModelLocatorConfig(
            activation=ModelLayerLocation.RESIDUAL_POST,
            location=[ModelLocation.LAYER, ModelLocation.POSITION],
        ),
        top_k=top_k,
    )


def _coordinates_config() -> LensConfig:
    return LensConfig(
        type=LensType.LOGIT_LENS,
        source=ModelLocatorConfig(
            coordinates=[
                {"activation": "resid.post", "layer": -2, "position": -1},
                {"activation": "mlp.out", "layer": -1, "position": -1},
            ]
        ),
    )


def _coordinates_config_of(
    *coordinates: dict,
    top_k: int = 1,
    lens_type: LensType = LensType.LOGIT_LENS,
    projection: ProjectionSpace | None = None,
) -> LensConfig:
    return LensConfig(
        type=lens_type,
        source=ModelLocatorConfig(coordinates=list(coordinates)),
        top_k=top_k,
        projection=projection,
    )


def _sliced(lens_input: LensInput, element: IndexElement) -> torch.Tensor:
    return lens_input.cache[element.transformer_activation()][element.slices()]


def _projected_weight(method, element: IndexElement) -> torch.Tensor:
    """A parameter lens reads its own tensor, so it needs no input to extract from."""
    return method.project(method.extract(None, element), element)


def _unembed(model, residual: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.ln_final(residual) @ model.W_U + model.b_U


def _element_token(index: IndexElement) -> int:
    return index.layer() * 100 + index[ModelLocation.POSITION]


class FakeLensMethod(AbstractLensMethod):
    """Plants fixed logit peaks at a per-element token so top-k selection is predictable."""

    PEAK_LOGIT = 10.0
    SECONDARY_LOGIT = 5.0

    calls: list[tuple[str, torch.Tensor]] = []

    def extract(self, lens_input, index):
        return lens_input.cache[index.transformer_activation()][index.slices()]

    def project(self, activation, index):
        type(self).calls.append((index.transformer_activation(), activation))

        logits = torch.zeros(activation.shape[0], self.model.cfg.d_vocab)
        logits[:, _element_token(index)] = self.PEAK_LOGIT
        logits[:, _element_token(index) + 1] = self.SECONDARY_LOGIT
        return logits


def _expected_probabilities(model) -> torch.Tensor:
    """Descending top probabilities of the fake logits: peak, secondary, then any zero-logit token."""
    logits = torch.zeros(model.cfg.d_vocab)
    logits[0] = FakeLensMethod.PEAK_LOGIT
    logits[1] = FakeLensMethod.SECONDARY_LOGIT
    return logits.softmax(dim=-1)


@pytest.fixture
def fake_method(monkeypatch) -> type[FakeLensMethod]:
    monkeypatch.setattr(LensMethodFactory, "get", staticmethod(lambda lens_type, activation: FakeLensMethod))
    FakeLensMethod.calls = []
    return FakeLensMethod


def test_top1_tokens_and_probabilities(model, fake_method):
    lens_input = LensInput(_declarative_config(), PROMPT)
    collector = Lens(model).apply(lens_input, return_collector=True, verbose=False)
    result = collector.result

    assert len(result) == model.cfg.n_layers * lens_input.tokens.shape[1]
    result_columns = [col for col in result.columns if not col.startswith(("source_", "index_"))]
    assert result_columns == ["token_0", "str_token_0", "probability_0"]

    expected = _expected_probabilities(model)
    for _, row in result.iterrows():
        token = row["source_layer"] * 100 + row["source_position"]
        assert row["token_0"] == token
        assert row["str_token_0"] == escape_token(model.to_single_str_token(token))
        assert row["probability_0"] == pytest.approx(expected[0].item(), rel=1e-5)


def test_top_k_ordered_by_probability(model, fake_method):
    lens_input = LensInput(_declarative_config(top_k=3), PROMPT)
    collector = Lens(model).apply(lens_input, return_collector=True, verbose=False)
    result = collector.result

    result_columns = [col for col in result.columns if not col.startswith(("source_", "index_"))]
    assert result_columns == [f"{name}_{i}" for i in range(3) for name in ("token", "str_token", "probability")]

    expected = _expected_probabilities(model)
    for _, row in result.iloc[[0, len(result) - 1]].iterrows():
        token = row["source_layer"] * 100 + row["source_position"]
        assert row["token_0"] == token
        assert row["token_1"] == token + 1
        assert [row["probability_0"], row["probability_1"], row["probability_2"]] == pytest.approx(
            expected[:3].tolist(), rel=1e-5
        )
        assert row["probability_0"] > row["probability_1"] > row["probability_2"]


def test_method_receives_sliced_cache_activations(model, fake_method):
    lens_input = LensInput(_declarative_config(), PROMPT)
    Lens(model).apply(lens_input, verbose=False)

    assert len(fake_method.calls) == len(lens_input.index)
    for (activation_name, activation), element in zip(fake_method.calls, lens_input.index, strict=True):
        position = element.source[ModelLocation.POSITION]
        assert activation_name == element.source.transformer_activation()
        assert torch.equal(activation, lens_input.cache[activation_name][:, position, :])


def test_coordinates_config(model, fake_method):
    lens_input = LensInput(_coordinates_config(), PROMPT)
    collector = Lens(model).apply(lens_input, return_collector=True, verbose=False)
    result = collector.result

    last_position = lens_input.tokens.shape[1] - 1
    assert len(result) == 2
    assert result["token_0"].tolist() == [
        (model.cfg.n_layers - 2) * 100 + last_position,
        (model.cfg.n_layers - 1) * 100 + last_position,
    ]


def test_artifacts_saved_to_disk(model, fake_method, tmp_path):
    config_yaml = tmp_path / "configs.yaml"
    config_yaml.write_text(yaml.safe_dump(LENS_YAML))
    config = ConfigFactory[LensConfig](str(config_yaml)).create("my_lens")

    save_dir = tmp_path / "lens_run"
    lens_input = LensInput(config, PROMPT)
    collector = Lens(model).apply(lens_input, save_dir=str(save_dir), return_collector=True, verbose=False)

    assert not (save_dir / "result.pt").exists()
    result_df = pd.read_csv(save_dir / "result.csv")
    assert len(result_df) == len(collector.result)
    assert result_df["token_0"].tolist() == collector.result["token_0"].tolist()
    assert result_df["str_token_1"].tolist() == collector.result["str_token_1"].tolist()
    assert result_df["probability_0"].tolist() == pytest.approx(collector.result["probability_0"].tolist())

    carriage_return = result_df[(result_df.source_layer == 2) & (result_df.source_position == 1)].iloc[0]
    assert carriage_return["token_0"] == 201
    assert carriage_return["str_token_0"] == "\\r"

    index_df = pd.read_csv(save_dir / "index.csv")
    assert len(index_df) == len(lens_input.index)

    with open(save_dir / "config.json", encoding="utf-8") as f:
        saved_config = json.load(f)
    assert saved_config["type"] == "logit_lens"
    assert saved_config["config_id"] == "my_lens"
    assert saved_config["top_k"] == 2


def test_logit_lens_projection_equals_model_logits(model):
    lens_input = LensInput(_declarative_config(), PROMPT).prepare(model)
    method = LogitLens(model, lens_input.config)
    element = next(iter(lens_input.index)).source

    final_residual = lens_input.cache[f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"]
    assert torch.allclose(method.project(final_residual, element), lens_input.logits, atol=1e-4)


def test_logit_lens_projects_a_single_mlp_neuron(model):
    layer = model.cfg.n_layers - 1
    config = _coordinates_config_of({"activation": "mlp.post", "layer": -1, "position": -1, "neuron": NEURON})
    lens_input = LensInput(config, PROMPT).prepare(model)
    element = next(iter(lens_input.index)).source

    activation = _sliced(lens_input, element)
    assert activation.shape == (1,)  # the neuron axis is dropped, leaving the batch

    logits = MLPLogitLens(model, config).project(activation, element)

    expected = _unembed(model, activation.unsqueeze(-1) * model.blocks[layer].mlp.W_out[NEURON])
    assert logits.shape == (1, model.cfg.d_vocab)
    assert torch.allclose(logits, expected, atol=1e-5)


def test_logit_lens_sums_a_list_of_mlp_neurons(model):
    layer = model.cfg.n_layers - 1
    neurons = [NEURON, NEURON + 1]
    config = _coordinates_config_of({"activation": "mlp.post", "layer": -1, "position": -1, "neuron": neurons})
    lens_input = LensInput(config, PROMPT).prepare(model)
    element = next(iter(lens_input.index)).source

    activation = _sliced(lens_input, element)
    assert activation.shape == (1, len(neurons))

    logits = MLPLogitLens(model, config).project(activation, element)

    w_out = model.blocks[layer].mlp.W_out
    contribution = sum(activation[:, i].unsqueeze(-1) * w_out[neuron] for i, neuron in enumerate(neurons))
    assert torch.allclose(logits, _unembed(model, contribution), atol=1e-5)


def test_logit_lens_over_all_mlp_neurons_is_mlp_out_without_its_bias(model):
    layer = model.cfg.n_layers - 1
    config = _coordinates_config_of(
        {"activation": "mlp.post", "layer": -1, "position": -1},
        {"activation": "mlp.out", "layer": -1, "position": -1},
    )
    lens_input = LensInput(config, PROMPT).prepare(model)
    hidden_element, out_element = (single_index.source for single_index in lens_input.index)

    logits = MLPLogitLens(model, config).project(_sliced(lens_input, hidden_element), hidden_element)

    # W_out lifts every neuron's contribution; the output bias belongs to no neuron and is left out.
    expected = _unembed(model, _sliced(lens_input, out_element) - model.blocks[layer].mlp.b_out)
    assert torch.allclose(logits, expected, atol=1e-4)


def test_neuron_lens_declarative_run(model):
    neurons = [NEURON, NEURON + 1]
    config = LensConfig(
        type=LensType.LOGIT_LENS,
        source=ModelLocatorConfig(
            activation=ModelLayerLocation.MLP_POST,
            location=[ModelLocation.LAYER, ModelLocation.NEURON, ModelLocation.POSITION],
            filter=Filter.from_config([{"neuron": neurons, "position": -1}]),
        ),
    )
    lens_input = LensInput(config, PROMPT)
    collector = Lens(model).apply(lens_input, return_collector=True, verbose=False)
    result = collector.result

    assert len(result) == model.cfg.n_layers * len(neurons)
    assert sorted(result["source_neuron"].unique()) == neurons
    assert result["source_position"].unique().tolist() == [lens_input.tokens.shape[1] - 1]

    method = MLPLogitLens(model, config)
    for single_index, (_, row) in zip(lens_input.index, result.iterrows(), strict=True):
        logits = method.project(_sliced(lens_input, single_index.source), single_index.source)
        assert row["token_0"] == logits.squeeze(0).argmax().item()


def test_attention_logit_lens_projects_a_single_head(model):
    layer = model.cfg.n_layers - 1
    config = _coordinates_config_of({"activation": "attn.z", "layer": -1, "position": -1, "head": HEAD})
    lens_input = LensInput(config, PROMPT).prepare(model)
    element = next(iter(lens_input.index)).source

    activation = _sliced(lens_input, element)
    assert activation.shape == (1, model.cfg.d_head)

    logits = AttentionLogitLens(model, config).project(activation, element)

    expected = _unembed(model, activation @ model.blocks[layer].attn.W_O[HEAD])
    assert logits.shape == (1, model.cfg.d_vocab)
    assert torch.allclose(logits, expected, atol=1e-5)


def test_attention_logit_lens_projects_a_single_head_dimension(model):
    layer = model.cfg.n_layers - 1
    config = _coordinates_config_of({"activation": "attn.z", "layer": -1, "position": -1, "head": HEAD, "neuron": 11})
    lens_input = LensInput(config, PROMPT).prepare(model)
    element = next(iter(lens_input.index)).source

    activation = _sliced(lens_input, element)
    assert activation.shape == (1,)  # both the head and the d_head axes are dropped

    logits = AttentionLogitLens(model, config).project(activation, element)

    expected = _unembed(model, activation.unsqueeze(-1) * model.blocks[layer].attn.W_O[HEAD, 11])
    assert torch.allclose(logits, expected, atol=1e-5)


def test_attention_logit_lens_sums_a_list_of_heads(model):
    layer = model.cfg.n_layers - 1
    heads = [HEAD, HEAD + 1]
    config = _coordinates_config_of({"activation": "attn.z", "layer": -1, "position": -1, "head": heads})
    lens_input = LensInput(config, PROMPT).prepare(model)
    element = next(iter(lens_input.index)).source

    activation = _sliced(lens_input, element)
    assert activation.shape == (1, len(heads), model.cfg.d_head)

    logits = AttentionLogitLens(model, config).project(activation, element)

    w_o = model.blocks[layer].attn.W_O
    contribution = sum(activation[:, i] @ w_o[head] for i, head in enumerate(heads))
    assert torch.allclose(logits, _unembed(model, contribution), atol=1e-5)


def test_attention_logit_lens_over_all_heads_is_attn_out_without_its_bias(model):
    layer = model.cfg.n_layers - 1
    config = _coordinates_config_of(
        {"activation": "attn.z", "layer": -1, "position": -1},
        {"activation": "attn.out", "layer": -1, "position": -1},
    )
    lens_input = LensInput(config, PROMPT).prepare(model)
    z_element, out_element = (single_index.source for single_index in lens_input.index)

    logits = AttentionLogitLens(model, config).project(_sliced(lens_input, z_element), z_element)

    expected = _unembed(model, _sliced(lens_input, out_element) - model.blocks[layer].attn.b_O)
    assert torch.allclose(logits, expected, atol=1e-4)


def test_attention_logit_lens_projects_values_through_the_same_route(model):
    layer = model.cfg.n_layers - 1
    config = _coordinates_config_of({"activation": "attn.v", "layer": -1, "position": -1, "head": HEAD})
    lens_input = LensInput(config, PROMPT).prepare(model)
    element = next(iter(lens_input.index)).source

    activation = _sliced(lens_input, element)
    logits = AttentionLogitLens(model, config).project(activation, element)

    expected = _unembed(model, activation @ model.blocks[layer].attn.W_O[HEAD])
    assert torch.allclose(logits, expected, atol=1e-5)


@pytest.mark.parametrize(("activation_name", "weight_name"), [("attn.q", "W_K"), ("attn.k", "W_Q")])
def test_qk_logit_lens_reads_against_the_embedding(model, activation_name, weight_name):
    layer = model.cfg.n_layers - 1
    config = _coordinates_config_of({"activation": activation_name, "layer": -1, "position": -1, "head": HEAD})
    lens_input = LensInput(config, PROMPT).prepare(model)
    element = next(iter(lens_input.index)).source

    activation = _sliced(lens_input, element)
    assert activation.shape == (1, model.cfg.d_head)

    logits = AttentionQKLogitLens(model, config).project(activation, element)

    with torch.no_grad():
        weight = getattr(model.blocks[layer].attn, weight_name)[HEAD]
        expected = activation @ weight.T @ model.W_E.T / model.cfg.attn_scale
    assert logits.shape == (1, model.cfg.d_vocab)
    assert torch.allclose(logits, expected, atol=1e-4)


def test_head_lens_declarative_run(model):
    config = LensConfig(
        type=LensType.LOGIT_LENS,
        source=ModelLocatorConfig(
            activation=ModelLayerLocation.ATTENTION_Z,
            location=[ModelLocation.LAYER, ModelLocation.HEAD, ModelLocation.POSITION],
            filter=Filter.from_config([{"position": -1}]),
        ),
    )
    lens_input = LensInput(config, PROMPT)
    collector = Lens(model).apply(lens_input, return_collector=True, verbose=False)
    result = collector.result

    assert len(result) == model.cfg.n_layers * model.cfg.n_heads
    assert sorted(result["source_head"].unique()) == list(range(model.cfg.n_heads))

    method = AttentionLogitLens(model, config)
    for single_index, (_, row) in zip(lens_input.index, result.iterrows(), strict=True):
        logits = method.project(_sliced(lens_input, single_index.source), single_index.source)
        assert row["token_0"] == logits.squeeze(0).argmax().item()


def test_negative_neuron_resolves_against_d_mlp(model):
    config = _coordinates_config_of({"activation": "mlp.post", "layer": -1, "position": -1, "neuron": -1})
    collector = Lens(model).apply(LensInput(config, PROMPT), return_collector=True, verbose=False)
    result = collector.result

    assert len(result) == 1
    assert result["source_neuron"].tolist() == [model.cfg.d_mlp - 1]
    assert 0 <= result["token_0"].iloc[0] < model.cfg.d_vocab


@pytest.fixture
def translators(model, monkeypatch):
    """Installs scaled-identity translators, so a translated residual stays hand-checkable."""
    requested = []

    def install(scale=lambda index: 0.0) -> list:
        built = torch.nn.ModuleList()
        for index in range(model.cfg.n_layers):
            linear = torch.nn.Linear(model.cfg.d_model, model.cfg.d_model)
            linear.weight.data = scale(index) * torch.eye(model.cfg.d_model)
            linear.bias.data = torch.zeros(model.cfg.d_model)
            built.append(linear)
        built.requires_grad_(False)

        def load(_, resource):
            requested.append(resource)
            return built

        monkeypatch.setattr(tuned_lens, "load_translators", load)
        return requested

    return install


@pytest.mark.parametrize(
    "coordinate",
    [
        {"activation": "embed", "position": -1},
        {"activation": "resid.post", "layer": 3, "position": -1},
        {"activation": "attn.out", "layer": 3, "position": -1},
        {"activation": "mlp.post", "layer": 3, "position": -1, "neuron": NEURON},
        {"activation": "attn.z", "layer": 3, "position": -1, "head": HEAD},
    ],
)
def test_tuned_lens_with_zero_translators_equals_the_logit_lens(model, translators, coordinate):
    translators()
    config = _coordinates_config_of(coordinate, lens_type=LensType.TUNED_LENS)
    lens_input = LensInput(config, PROMPT).prepare(model)
    element = next(iter(lens_input.index)).source
    activation = _sliced(lens_input, element)

    tuned = LensMethodFactory.create(config, model, element.activation()).project(activation, element)
    logit = LensMethodFactory.create(_coordinates_config_of(coordinate), model, element.activation()).project(
        activation, element
    )

    assert torch.allclose(tuned, logit, atol=1e-5)


@pytest.mark.parametrize(
    ("coordinate", "translator_index"),
    [
        ({"activation": "resid.pre", "layer": 3, "position": -1}, 3),
        ({"activation": "resid.post", "layer": 3, "position": -1}, 4),
        ({"activation": "resid.mid", "layer": 3, "position": -1}, 4),
    ],
)
def test_tuned_lens_translates_the_residual_of_its_state(model, translators, coordinate, translator_index):
    translators(scale=lambda index: index / 10)
    config = _coordinates_config_of(coordinate, lens_type=LensType.TUNED_LENS)
    lens_input = LensInput(config, PROMPT).prepare(model)
    element = next(iter(lens_input.index)).source
    activation = _sliced(lens_input, element)

    logits = TunedLens(model, config).project(activation, element)

    assert torch.allclose(logits, _unembed(model, (1 + translator_index / 10) * activation), atol=1e-4)


def test_tuned_lens_leaves_the_last_state_untranslated(model, translators):
    translators(scale=lambda index: index / 10)
    config = _coordinates_config_of(
        {"activation": "resid.post", "layer": -1, "position": -1}, lens_type=LensType.TUNED_LENS
    )
    lens_input = LensInput(config, PROMPT).prepare(model)
    element = next(iter(lens_input.index)).source

    logits = TunedLens(model, config).project(_sliced(lens_input, element), element)

    assert torch.allclose(logits, lens_input.logits[:, -1], atol=1e-4)


def test_tuned_lens_declarative_run(model, translators):
    requested = translators(scale=lambda index: index / 10)
    config = LensConfig(
        type=LensType.TUNED_LENS,
        translators="gpt2",
        source=ModelLocatorConfig(
            activation=ModelLayerLocation.RESIDUAL_POST,
            location=[ModelLocation.LAYER, ModelLocation.POSITION],
            filter=Filter.from_config([{"position": -1}]),
        ),
    )
    lens_input = LensInput(config, PROMPT)
    result = Lens(model).apply(lens_input, return_collector=True, verbose=False).result

    assert len(result) == model.cfg.n_layers
    assert requested == ["gpt2"] * model.cfg.n_layers

    method = TunedLens(model, config)
    for single_index, (_, row) in zip(lens_input.index, result.iterrows(), strict=True):
        logits = method.project(_sliced(lens_input, single_index.source), single_index.source)
        assert row["token_0"] == logits.squeeze(0).argmax().item()


def test_logit_lens_top1_matches_model_prediction_at_final_layer(model):
    lens_input = LensInput(_declarative_config(), PROMPT)
    collector = Lens(model).apply(lens_input, return_collector=True, verbose=False)
    result = collector.result

    probabilities = lens_input.logits.softmax(dim=-1)[0]
    for _, row in result[result.source_layer == model.cfg.n_layers - 1].iterrows():
        position = row["source_position"]
        assert row["token_0"] == probabilities[position].argmax().item()
        assert row["str_token_0"] == escape_token(model.to_single_str_token(row["token_0"]))
        assert row["probability_0"] == pytest.approx(probabilities[position].max().item(), rel=1e-4)


PARAMETER_LENS_YAML = {
    "my_parameter_lens": {
        "type": "parameter_lens",
        "top_k": 2,
        "source": {
            "activation": "mlp.post",
            "location": ["layer", "neuron"],
            "filter": [{"neuron": NEURON}],
        },
    }
}


def _parameter_config(top_k: int = 1, projection: ProjectionSpace | None = None) -> LensConfig:
    return LensConfig(
        type=LensType.PARAMETER_LENS,
        top_k=top_k,
        projection=projection,
        source=ModelLocatorConfig(
            activation=ModelLayerLocation.MLP_POST,
            location=[ModelLocation.LAYER, ModelLocation.NEURON],
            filter=Filter(inclusion_terms=[{ModelLocation.NEURON: NEURON}]),
        ),
    )


def _parameter_element(model, coordinate: dict, projection: ProjectionSpace | None = None) -> IndexElement:
    config = _coordinates_config_of(coordinate, lens_type=LensType.PARAMETER_LENS, projection=projection)
    return next(iter(LensInput(config).prepare(model).index)).source


def test_parameter_lens_needs_no_prompt(model):
    collector = Lens(model).apply(LensInput(_parameter_config()), return_collector=True, verbose=False)
    result = collector.result

    assert len(result) == model.cfg.n_layers
    assert result["source_neuron"].tolist() == [NEURON] * model.cfg.n_layers
    result_columns = [col for col in result.columns if not col.startswith(("source_", "index_"))]
    assert result_columns == ["token_0", "str_token_0", "probability_0"]


def test_mlp_value_vector_is_the_row_of_w_out_through_the_unembedding(model):
    method = MLPValueParameterLens(model, _parameter_config())
    element = _parameter_element(model, {"activation": "mlp.post", "layer": LAYER, "neuron": NEURON})

    assert torch.equal(_projected_weight(method, element), model.blocks[LAYER].mlp.W_out[NEURON] @ model.W_U)


def test_parameter_lens_projection_embedding_reads_the_same_vector_through_w_e(model):
    method = MLPValueParameterLens(model, _parameter_config(projection=ProjectionSpace.EMBEDDING))
    weight = model.blocks[LAYER].mlp.W_out[NEURON]
    element = _parameter_element(
        model, {"activation": "mlp.post", "layer": LAYER, "neuron": NEURON}, projection=ProjectionSpace.EMBEDDING
    )

    projected = _projected_weight(method, element)

    assert torch.equal(projected, weight @ model.W_E.T)
    # TransformerLens centers the unembedding and folds LayerNorm in, so the two spaces differ even
    # though GPT-2 ties its embedding and unembedding.
    assert not torch.equal(projected, weight @ model.W_U)


def test_mlp_key_vector_is_the_column_of_w_in(model):
    method = MLPKeyParameterLens(model, _parameter_config())
    element = _parameter_element(model, {"activation": "mlp.pre", "layer": LAYER, "neuron": NEURON})

    assert torch.equal(_projected_weight(method, element), model.blocks[LAYER].mlp.W_in[:, NEURON] @ model.W_U)


@pytest.mark.parametrize(("activation_name", "weight_name"), [("attn.q", "W_Q"), ("attn.k", "W_K"), ("attn.v", "W_V")])
def test_attention_subhead_read_vector_is_the_column_of_its_weight(model, activation_name, weight_name):
    method = AttentionReadParameterLens(model, _parameter_config())
    element = _parameter_element(model, {"activation": activation_name, "layer": LAYER, "head": HEAD, "neuron": 12})

    weight = getattr(model.blocks[LAYER].attn, weight_name)
    assert torch.equal(_projected_weight(method, element), weight[HEAD, :, 12] @ model.W_U)


def test_attention_subhead_write_vector_is_the_row_of_w_o(model):
    method = AttentionWriteParameterLens(model, _parameter_config())
    element = _parameter_element(model, {"activation": "attn.z", "layer": LAYER, "head": HEAD, "neuron": 12})

    assert torch.equal(_projected_weight(method, element), model.blocks[LAYER].attn.W_O[HEAD, 12] @ model.W_U)


def test_mlp_value_vectors_of_grouped_neurons_are_summed(model):
    method = MLPValueParameterLens(model, _parameter_config())
    neurons = [NEURON, NEURON + 1]
    element = _parameter_element(model, {"activation": "mlp.post", "layer": LAYER, "neuron": neurons})

    w_out = model.blocks[LAYER].mlp.W_out
    expected = (w_out[NEURON] + w_out[NEURON + 1]) @ model.W_U
    assert torch.allclose(_projected_weight(method, element), expected, atol=1e-5)


def test_mlp_value_vector_without_a_neuron_sums_every_neuron(model):
    method = MLPValueParameterLens(model, _parameter_config())
    element = _parameter_element(model, {"activation": "mlp.post", "layer": LAYER})

    expected = model.blocks[LAYER].mlp.W_out.sum(0) @ model.W_U
    assert torch.allclose(_projected_weight(method, element), expected, atol=1e-4)


def test_mlp_key_vectors_of_grouped_neurons_are_summed(model):
    method = MLPKeyParameterLens(model, _parameter_config())
    neurons = [NEURON, NEURON + 1]
    element = _parameter_element(model, {"activation": "mlp.pre", "layer": LAYER, "neuron": neurons})

    w_in = model.blocks[LAYER].mlp.W_in
    expected = (w_in[:, NEURON] + w_in[:, NEURON + 1]) @ model.W_U
    assert torch.allclose(_projected_weight(method, element), expected, atol=1e-5)


def test_attention_read_vectors_of_grouped_heads_and_neurons_are_summed(model):
    method = AttentionReadParameterLens(model, _parameter_config())
    heads, neurons = [HEAD, HEAD + 1], [12, 13]
    element = _parameter_element(model, {"activation": "attn.q", "layer": LAYER, "head": heads, "neuron": neurons})

    w_q = model.blocks[LAYER].attn.W_Q
    expected = sum(w_q[head, :, neuron] for head in heads for neuron in neurons) @ model.W_U
    assert torch.allclose(_projected_weight(method, element), expected, atol=1e-5)


def test_attention_write_vectors_of_grouped_heads_and_neurons_are_summed(model):
    method = AttentionWriteParameterLens(model, _parameter_config())
    heads, neurons = [HEAD, HEAD + 1], [12, 13]
    element = _parameter_element(model, {"activation": "attn.z", "layer": LAYER, "head": heads, "neuron": neurons})

    w_o = model.blocks[LAYER].attn.W_O
    expected = sum(w_o[head, neuron] for head in heads for neuron in neurons) @ model.W_U
    assert torch.allclose(_projected_weight(method, element), expected, atol=1e-5)


def test_attention_write_vector_without_a_head_sums_every_head(model):
    method = AttentionWriteParameterLens(model, _parameter_config())
    element = _parameter_element(model, {"activation": "attn.z", "layer": LAYER, "neuron": 12})

    expected = model.blocks[LAYER].attn.W_O[:, 12].sum(0) @ model.W_U
    assert torch.allclose(_projected_weight(method, element), expected, atol=1e-5)


def test_parameter_lens_grouped_neurons_collect_a_single_row(model):
    neurons = [12, 21]
    config = _coordinates_config_of(
        {"activation": "mlp.post", "layer": LAYER, "neuron": neurons}, lens_type=LensType.PARAMETER_LENS
    )
    collector = Lens(model).apply(LensInput(config), return_collector=True, verbose=False)
    result = collector.result

    w_out = model.blocks[LAYER].mlp.W_out
    assert len(result) == 1
    assert result.iloc[0]["source_neuron"] == neurons
    assert result.iloc[0]["token_0"] == ((w_out[12] + w_out[21]) @ model.W_U).argmax().item()


def test_parameter_lens_top_k_tokens_come_from_the_projected_weight(model):
    collector = Lens(model).apply(LensInput(_parameter_config(top_k=3)), return_collector=True, verbose=False)
    result = collector.result

    expected = (model.blocks[LAYER].mlp.W_out[NEURON] @ model.W_U).softmax(dim=-1).topk(3)
    row = result[result.source_layer == LAYER].iloc[0]
    assert [row["token_0"], row["token_1"], row["token_2"]] == expected.indices.tolist()
    assert [row["str_token_0"], row["str_token_1"]] == [
        model.to_single_str_token(token) for token in expected.indices.tolist()[:2]
    ]
    assert row["probability_0"] == pytest.approx(expected.values[0].item(), rel=1e-5)


def test_parameter_lens_artifacts_saved_to_disk(model, tmp_path):
    config_yaml = tmp_path / "configs.yaml"
    config_yaml.write_text(yaml.safe_dump(PARAMETER_LENS_YAML))
    config = ConfigFactory[LensConfig](str(config_yaml)).create("my_parameter_lens")

    save_dir = tmp_path / "parameter_lens_run"
    parameter_input = LensInput(config)
    collector = Lens(model).apply(parameter_input, save_dir=str(save_dir), return_collector=True, verbose=False)

    result_df = pd.read_csv(save_dir / "result.csv")
    assert result_df["token_0"].tolist() == collector.result["token_0"].tolist()
    assert result_df["probability_1"].tolist() == pytest.approx(collector.result["probability_1"].tolist())

    index_df = pd.read_csv(save_dir / "index.csv")
    assert len(index_df) == len(parameter_input.index)

    with open(save_dir / "config.json", encoding="utf-8") as f:
        saved_config = json.load(f)
    assert saved_config["type"] == "parameter_lens"
    assert saved_config["config_id"] == "my_parameter_lens"


def test_parameter_lens_config_with_a_prompt_raises():
    with pytest.raises(ValueError, match="parameter lens .* takes no prompt"):
        LensInput(_parameter_config(), PROMPT)


def test_activation_lens_config_without_a_prompt_raises():
    with pytest.raises(ValueError, match="'logit_lens' reads activations, so it needs a prompt"):
        LensInput(_declarative_config())
