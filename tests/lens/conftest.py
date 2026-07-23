import pandas as pd
import pytest
import torch

from intex.lens.collector import LensCollector, escape_token
from intex.lens.config import LensConfig, LensType, ProjectionSpace
from intex.lens.input import LensInput
from intex.lens.lens import Lens
from intex.model.config import ModelLayerLocation, ModelLocation, ModelLocatorConfig

PROMPT = "The capital of France is"
NEURON = 373
HEAD = 7
LAYER = 8


def coordinates_config(
    *coordinates: dict,
    lens_type: LensType = LensType.LOGIT_LENS,
    top_k: int = 1,
    projection: ProjectionSpace | None = None,
) -> LensConfig:
    return LensConfig(
        type=lens_type,
        top_k=top_k,
        projection=projection,
        source=ModelLocatorConfig(coordinates=list(coordinates)),
    )


def declarative_config(
    activation: ModelLayerLocation,
    location: list[ModelLocation],
    lens_type: LensType = LensType.LOGIT_LENS,
    top_k: int = 1,
    **source,
) -> LensConfig:
    return LensConfig(
        type=lens_type,
        top_k=top_k,
        source=ModelLocatorConfig(activation=activation, location=location, **source),
    )


def run_lens(model, config: LensConfig, prompt: str | None = PROMPT, save_dir=None) -> LensCollector:
    prompt = None if config.type is LensType.PARAMETER_LENS else prompt
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


def unembed(model, residual: torch.Tensor) -> torch.Tensor:
    """The model's own read-out of a residual-stream vector."""
    with torch.no_grad():
        return model.ln_final(residual) @ model.W_U + model.b_U


def assert_row_reads(model, row: pd.Series, logits: torch.Tensor, top_k: int = 1) -> None:
    """The row's top-k columns are the top-k of ``logits`` — token, escaped string and probability."""
    probabilities, tokens = logits.detach().flatten().softmax(dim=-1).topk(top_k)

    for i, (token, probability) in enumerate(zip(tokens.tolist(), probabilities.tolist(), strict=True)):
        assert row[f"token_{i}"] == token
        assert row[f"str_token_{i}"] == escape_token(model.to_single_str_token(token))
        assert row[f"probability_{i}"] == pytest.approx(probability, rel=1e-4)


@pytest.fixture(scope="module")
def activations(model):
    """The prompt's activations from a plain forward pass — what a run's projections are checked against."""
    with torch.no_grad():
        logits, cache = model.run_with_cache(model.to_tokens(PROMPT))
    return logits, cache
