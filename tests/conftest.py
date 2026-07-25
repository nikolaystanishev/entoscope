from functools import lru_cache

import pytest
import torch
import yaml
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
)

from intex.operation.config import AbstractConfig, ConfigFactory

try:
    from transformer_lens.model_bridge import TransformerBridge
except ImportError:  # transformer-lens 2.x (Intel macOS fallback)
    TransformerBridge = None

# Every model fixture runs its tests once per backend. The bridge is what intex is built around;
# HookedTransformer stays supported for the platforms that cannot install a transformer-lens with it.
BACKENDS = ["hooked"] if TransformerBridge is None else ["bridge", "hooked"]

# Shared by both backends' tiny models, so an index means the same thing whichever one is under test.
# ``d_model`` is ``n_heads * d_head`` because a HuggingFace config derives the head size from the
# two rather than carrying it, and a bridge reads its ``cfg`` back from there.
TINY = {"n_layers": 3, "d_model": 16, "n_ctx": 16, "d_head": 4, "n_heads": 4, "d_mlp": 6, "d_vocab": 50257}


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
    settings = {**TINY, "act_fn": "gelu", "seed": 0, **overrides}
    model = HookedTransformer(HookedTransformerConfig(**settings), tokenizer=_tokenizer())
    model.eval()
    return model


def build_tiny_bridge(hf_model):
    """The bridge counterpart of ``build_tiny_model``.

    A bridge wraps a HuggingFace model instead of a ``HookedTransformerConfig``, so the architecture
    variants come from choosing the HF class rather than from setting a flag. The wrapped model is
    evaluated separately: a bridge holds it outside ``_modules``, so ``eval()`` does not reach it and
    a freshly constructed HuggingFace model would keep dropout on.
    """
    hf_model.eval()
    model = TransformerBridge.boot_transformers("tiny", hf_model=hf_model, tokenizer=_tokenizer(), device="cpu")
    model.eval()
    return model


def _tiny_hf(factory, config_class, **settings):
    torch.manual_seed(0)
    return factory(
        config_class(
            num_hidden_layers=TINY["n_layers"],
            hidden_size=TINY["d_model"],
            num_attention_heads=TINY["n_heads"],
            intermediate_size=TINY["d_mlp"],
            vocab_size=TINY["d_vocab"],
            max_position_embeddings=TINY["n_ctx"],
            **settings,
        )
    )


def _tiny_gpt2():
    torch.manual_seed(0)
    return GPT2LMHeadModel(
        GPT2Config(
            n_layer=TINY["n_layers"],
            n_embd=TINY["d_model"],
            n_head=TINY["n_heads"],
            n_inner=TINY["d_mlp"],
            n_positions=TINY["n_ctx"],
            vocab_size=TINY["d_vocab"],
        )
    )


def _gpt2_bridge():
    model = TransformerBridge.boot_transformers("gpt2", device="cpu")
    model.eval()
    return model


def _gpt2_hooked():
    model = HookedTransformer.from_pretrained("gpt2")
    model.eval()
    return model


@pytest.fixture(scope="session", params=BACKENDS)
def backend(request) -> str:
    """Which transformer_lens wrapper the model fixtures build.

    Every model fixture derives from this one, so a test that takes more than one of them gets
    them all on the same backend. A test reaching a fixture through ``request.getfixturevalue``
    has to name this fixture too, since that is what carries the parameter.
    """
    return request.param


def _backend_fixture(bridge, hooked):
    @pytest.fixture(scope="session")
    def fixture(backend):
        return bridge() if backend == "bridge" else hooked()

    return fixture


model = _backend_fixture(_gpt2_bridge, _gpt2_hooked)

small_model = _backend_fixture(lambda: build_tiny_bridge(_tiny_gpt2()), build_tiny_model)

# Llama is gated as well as grouped-query; the tests using this fixture only read head counts.
gqa_model = _backend_fixture(
    lambda: build_tiny_bridge(_tiny_hf(LlamaForCausalLM, LlamaConfig, num_key_value_heads=2)),
    lambda: build_tiny_model(n_key_value_heads=2),
)

gated_model = _backend_fixture(
    lambda: build_tiny_bridge(_tiny_hf(LlamaForCausalLM, LlamaConfig, num_key_value_heads=TINY["n_heads"])),
    lambda: build_tiny_model(gated_mlp=True, act_fn="silu"),
)

parallel_model = _backend_fixture(
    lambda: build_tiny_bridge(_tiny_hf(GPTNeoXForCausalLM, GPTNeoXConfig, use_parallel_residual=True)),
    lambda: build_tiny_model(parallel_attn_mlp=True),
)


def forbid_model_runs(monkeypatch, model) -> None:
    """Make every forward pass through ``model`` fail — what an operation's plan must never need.

    Patched on the instance, so the session-scoped model is restored when the test's monkeypatch
    unwinds. Tokenization and ``cfg`` stay untouched: resolving an index is allowed to read those.
    """

    def fail(*args, **kwargs):
        raise AssertionError("the model was run")

    for name in ("run_with_cache", "run_with_hooks", "forward"):
        monkeypatch.setattr(model, name, fail, raising=False)


@pytest.fixture
def config_from_yaml(tmp_path):
    """Source a config the way a run does: written to a YAML file and loaded through ConfigFactory."""

    def create(config_class: type[AbstractConfig], config_id: str, raw: dict, file_name: str = "configs.yaml"):
        path = tmp_path / file_name
        path.write_text(yaml.safe_dump({config_id: raw}))
        return ConfigFactory[config_class](str(path)).create(config_id)

    return create
