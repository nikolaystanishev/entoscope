from functools import lru_cache

import pytest
import yaml
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformers import AutoTokenizer

from intex.operation.config import AbstractConfig, ConfigFactory


@pytest.fixture(scope="session")
def model() -> HookedTransformer:
    model = HookedTransformer.from_pretrained("gpt2")
    model.eval()
    return model


@lru_cache(maxsize=1)
def _tokenizer():
    """Resolved once: naming it on the config would send every tiny model to the hub for its own copy."""
    return AutoTokenizer.from_pretrained("gpt2")


def build_tiny_model(**overrides) -> HookedTransformer:
    """A real transformer with random weights, small enough to enumerate a whole location axis.

    GPT-2 answers "what does this compute", these answer "what does the config index" — and they
    reach architectures GPT-2 does not have (grouped-query attention, a gated MLP, parallel
    attention/MLP). The tokenizer is GPT-2's, so prompts tokenize the same way.
    """
    settings = {
        "n_layers": 3,
        "d_model": 8,
        "n_ctx": 16,
        "d_head": 4,
        "n_heads": 4,
        "d_mlp": 6,
        "d_vocab": 50257,
        "act_fn": "gelu",
        "seed": 0,
        **overrides,
    }
    model = HookedTransformer(HookedTransformerConfig(**settings), tokenizer=_tokenizer())
    model.eval()
    return model


@pytest.fixture(scope="session")
def small_model() -> HookedTransformer:
    return build_tiny_model()


@pytest.fixture(scope="session")
def gqa_model() -> HookedTransformer:
    return build_tiny_model(n_key_value_heads=2)


@pytest.fixture(scope="session")
def gated_model() -> HookedTransformer:
    return build_tiny_model(gated_mlp=True, act_fn="silu")


@pytest.fixture(scope="session")
def parallel_model() -> HookedTransformer:
    return build_tiny_model(parallel_attn_mlp=True)


@pytest.fixture
def config_from_yaml(tmp_path):
    """Source a config the way a run does: written to a YAML file and loaded through ConfigFactory."""

    def create(config_class: type[AbstractConfig], config_id: str, raw: dict, file_name: str = "configs.yaml"):
        path = tmp_path / file_name
        path.write_text(yaml.safe_dump({config_id: raw}))
        return ConfigFactory[config_class](str(path)).create(config_id)

    return create
