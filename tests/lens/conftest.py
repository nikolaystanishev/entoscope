import pandas as pd
import pytest
import torch

from entoscope.lens.collector import LensCollector
from entoscope.lens.config import LensConfig, LensMethodConfig
from entoscope.lens.input import LensInput
from entoscope.lens.lens import Lens
from entoscope.lens.method.factory import LensMethodFactory
from entoscope.model.config import Coordinates, ModelAxis, ModelLayerLocation, ModelLocatorConfig
from entoscope.model.token import escape_token

PROMPT = "The capital of France is"
NEURON = 373
HEAD = 7
LAYER = 8


def coordinates_config(*coordinates: dict, method: LensMethodConfig | None = None, top_k: int = 1) -> LensConfig:
    return LensConfig(
        method=method or LensMethodConfig(type="logit_lens"),
        top_k=top_k,
        source=ModelLocatorConfig(coordinates=Coordinates.from_config(list(coordinates))),
    )


def declarative_config(
    activation: ModelLayerLocation,
    location: list[ModelAxis],
    method: LensMethodConfig | None = None,
    top_k: int = 1,
    **source,
) -> LensConfig:
    return LensConfig(
        method=method or LensMethodConfig(type="logit_lens"),
        top_k=top_k,
        source=ModelLocatorConfig(activation=activation, location=location, **source),
    )


def run_lens(model, config: LensConfig, prompt: str | None = PROMPT, save_dir=None) -> LensCollector:
    prompt = prompt if LensMethodFactory.get(config.method.type).reads_activations else None
    lens_input = LensInput(config, prompt)
    collector = Lens(model).apply(
        lens_input, save_dir=None if save_dir is None else str(save_dir), return_collector=True, verbose=False
    )
    assert collector is not None
    return collector


def lens_rows(model, config: LensConfig, prompt: str | None = PROMPT) -> pd.DataFrame:
    return run_lens(model, config, prompt).result


def single_row(model, config: LensConfig, prompt: str | None = PROMPT) -> pd.Series:
    result = lens_rows(model, config, prompt)
    assert len(result) == 1
    return result.iloc[0]


def mlp_matrix(model, layer: int, attribute: str) -> torch.Tensor:
    """A block's MLP weight in HookedTransformer's orientation, read without the reader under test.

    ``entoscope.model.weight.model_weight`` is what a lens reads, so an expectation taken from it would
    only restate the run. This reads the parameter the backend stores and settles the orientation by
    its shape - ``W_in``/``W_gate`` [d_model, d_mlp], ``W_out`` [d_mlp, d_model] - which no fixture
    here leaves ambiguous, none of them having ``d_model == d_mlp``.
    """
    weight = getattr(model.blocks[layer].mlp, attribute)
    rows = model.cfg.d_mlp if attribute == "W_out" else model.cfg.d_model
    return weight if weight.shape[0] == rows else weight.T


def unembed(model, residual: torch.Tensor) -> torch.Tensor:
    """The model's own read-out of a residual-stream vector."""
    with torch.no_grad():
        return model.ln_final(residual) @ model.W_U + model.b_U


def assert_row_reads(model, row: pd.Series, logits: torch.Tensor, top_k: int = 1) -> None:
    """The row's top-k columns are the top-k of ``logits`` - token, escaped string and probability."""
    probabilities, tokens = logits.detach().flatten().softmax(dim=-1).topk(top_k)

    for i, (token, probability) in enumerate(zip(tokens.tolist(), probabilities.tolist(), strict=True)):
        assert row[f"token_{i}"] == token
        assert row[f"str_token_{i}"] == escape_token(model.to_single_str_token(token))
        assert row[f"probability_{i}"] == pytest.approx(probability, rel=1e-4)


@pytest.fixture(scope="module")
def activations(model):
    """The prompt's activations from a plain forward pass - what a run's projections are checked against."""
    with torch.no_grad():
        logits, cache = model.run_with_cache(model.to_tokens(PROMPT))
    return logits, cache
