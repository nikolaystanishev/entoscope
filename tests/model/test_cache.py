import torch

from intex.lens.config import LensConfig, LensType
from intex.lens.input import LensInput
from intex.lens.lens import Lens
from intex.model.cache import ActivationDiskCache, CachedModel, _normalize_names_filter
from intex.model.config import ModelLayerLocation, ModelLocation, ModelLocatorConfig

PROMPT = "Paris is located in"
FINGERPRINT = "fingerprint-a"


def _key(cache: ActivationDiskCache, *, fingerprint=FINGERPRINT, model_args=None, names=None, extra=None) -> str:
    return cache.key(
        model_fingerprint=fingerprint,
        model_args=model_args if model_args is not None else (torch.tensor([[1, 2, 3]]),),
        names_filter=_normalize_names_filter(names),
        extra=extra or {},
    )


# --- Key derivation (no model needed) ---------------------------------------------------------


def test_key_is_deterministic(tmp_path):
    cache = ActivationDiskCache(str(tmp_path))
    assert _key(cache) == _key(cache)


def test_key_ignores_names_filter_order(tmp_path):
    cache = ActivationDiskCache(str(tmp_path))
    assert _key(cache, names=["b", "a"]) == _key(cache, names=["a", "b"])


def test_key_distinguishes_relevant_inputs(tmp_path):
    cache = ActivationDiskCache(str(tmp_path))
    base = _key(cache)
    assert _key(cache, model_args=(torch.tensor([[1, 2, 4]]),)) != base  # different tokens
    assert _key(cache, names=["a"]) != base  # different filter (base is "all")
    assert _key(cache, names=["a"]) != _key(cache, names=["a", "b"])  # subset differs
    assert _key(cache, fingerprint="fingerprint-b") != base  # different model
    assert _key(cache, extra={"remove_batch_dim": True}) != base  # output-shaping kwarg
    # Tensor-valued model-input kwargs (e.g. attention_mask) must not collide, even at sizes where
    # ``repr`` would truncate the tensor to identical text.
    mask_a = torch.ones(1, 40, dtype=torch.long)
    mask_b = mask_a.clone()
    mask_b[0, -1] = 0
    assert _key(cache, extra={"attention_mask": mask_a}) != base
    assert _key(cache, extra={"attention_mask": mask_a}) != _key(cache, extra={"attention_mask": mask_b})


def test_normalize_names_filter():
    assert _normalize_names_filter(None) == "*ALL*"
    assert _normalize_names_filter("x") == ["x"]
    assert _normalize_names_filter(["b", "a"]) == ["a", "b"]


# --- Round-trip against the real model --------------------------------------------------------


def _count_forward_passes(model, monkeypatch) -> dict:
    calls = {"n": 0}
    original = model.run_with_cache

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "run_with_cache", counting)
    return calls


def test_second_call_is_served_from_disk(model, tmp_path, monkeypatch):
    calls = _count_forward_passes(model, monkeypatch)
    wrapped = CachedModel(model, str(tmp_path))
    tokens = model.to_tokens(PROMPT)
    names = ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"]

    logits1, cache1 = wrapped.run_with_cache(tokens, names_filter=names)
    assert calls["n"] == 1
    assert list(tmp_path.rglob("*.pt"))  # a cache file was written

    logits2, cache2 = wrapped.run_with_cache(tokens, names_filter=names)
    assert calls["n"] == 1  # no recomputation on the hit

    assert torch.equal(logits1, logits2)
    assert set(cache1.keys()) == set(cache2.keys()) == set(names)
    for name in names:
        assert torch.equal(cache1[name], cache2[name])


def test_cached_result_matches_direct_run(model, tmp_path):
    wrapped = CachedModel(model, str(tmp_path))
    tokens = model.to_tokens("The capital of France is")
    names = ["blocks.5.hook_resid_post"]

    direct_logits, direct_cache = model.run_with_cache(tokens, names_filter=names)
    wrapped.run_with_cache(tokens, names_filter=names)  # populate the cache
    cached_logits, cached_cache = wrapped.run_with_cache(tokens, names_filter=names)  # read from disk

    assert torch.equal(direct_logits, cached_logits)
    assert torch.equal(direct_cache[names[0]], cached_cache[names[0]])


def test_unfiltered_run_is_cached(model, tmp_path, monkeypatch):
    calls = _count_forward_passes(model, monkeypatch)
    wrapped = CachedModel(model, str(tmp_path))
    tokens = model.to_tokens(PROMPT)

    logits1, cache1 = wrapped.run_with_cache(tokens)  # circuit-style, no names_filter
    logits2, cache2 = wrapped.run_with_cache(tokens)
    assert calls["n"] == 1
    assert set(cache1.keys()) == set(cache2.keys())
    assert torch.equal(logits1, logits2)


def test_pass_through_without_cache_dir(model):
    wrapped = CachedModel(model, cache_dir=None)
    tokens = model.to_tokens("Hello world")
    names = ["blocks.0.hook_resid_post"]

    logits, cache = wrapped.run_with_cache(tokens, names_filter=names)
    direct_logits, direct_cache = model.run_with_cache(tokens, names_filter=names)
    assert torch.equal(logits, direct_logits)
    assert torch.equal(cache[names[0]], direct_cache[names[0]])


def test_forwards_model_attributes(model):
    wrapped = CachedModel(model, cache_dir=None)
    assert wrapped.cfg is model.cfg
    assert wrapped.W_U is model.W_U
    assert callable(wrapped.run_with_hooks)
    assert torch.equal(wrapped.to_tokens("Hello world"), model.to_tokens("Hello world"))


# --- Works transparently for a real operation -------------------------------------------------


def _lens_config() -> LensConfig:
    return LensConfig(
        type=LensType.LOGIT_LENS,
        source=ModelLocatorConfig(
            activation=ModelLayerLocation.RESIDUAL_POST,
            location=[ModelLocation.LAYER, ModelLocation.POSITION],
        ),
    )


def test_lens_operation_reuses_cached_activations(model, tmp_path, monkeypatch):
    calls = _count_forward_passes(model, monkeypatch)
    wrapped = CachedModel(model, str(tmp_path))

    Lens(wrapped).apply(LensInput(_lens_config(), PROMPT), return_collector=True, verbose=False)
    assert calls["n"] == 1
    assert list(tmp_path.rglob("*.pt"))

    # A second run over the same prompt/model reads its activations from disk.
    Lens(wrapped).apply(LensInput(_lens_config(), PROMPT), return_collector=True, verbose=False)
    assert calls["n"] == 1
