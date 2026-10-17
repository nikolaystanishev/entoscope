"""``CachedModel.run_with_cache`` - what it returns, what it writes, and when it recomputes.

A cache hit is proven without a spy: the model's weights are changed between two calls, and the call
served from disk keeps returning the first call's tensors while the unwrapped model follows the new
weights. Which requests share an entry is read off the cache directory - one file per distinct key.
"""

import pathlib

import pytest
import torch

from entoscope.model.cache import CachedModel

from ..conftest import build_small_model

PROMPT = "Paris is located in"
OTHER_PROMPT = "Rome is located in"
NAMES = ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"]


@pytest.fixture
def wrapped(backend, tmp_path):
    """A real tiny model behind the disk cache, plus the plain model to compare against."""
    return CachedModel(build_small_model(backend), str(tmp_path))


def cache_files(cache_dir) -> list[pathlib.Path]:
    return sorted(cache_dir.rglob("*.pt"))


def tokens_of(model, prompt: str = PROMPT) -> torch.Tensor:
    return model.to_tokens(prompt)


def bump_weights(model) -> None:
    """Change what a forward pass computes, without changing the model's shape (its cache key)."""
    with torch.no_grad():
        model.blocks[0].mlp.W_out.mul_(3.0)
        model.blocks[0].mlp.b_out.add_(1.0)


def assert_same_cache(left, right, names=NAMES) -> None:
    assert set(left.keys()) == set(right.keys())
    for name in names:
        assert torch.equal(left[name], right[name])


# --------------------------------------------------------------------------------------------------
# The call itself
# --------------------------------------------------------------------------------------------------
def test_the_first_call_returns_what_the_unwrapped_model_returns(wrapped, tmp_path):
    tokens = tokens_of(wrapped)

    logits, cache = wrapped.run_with_cache(tokens, names_filter=NAMES)
    direct_logits, direct_cache = wrapped._model.run_with_cache(tokens, names_filter=NAMES)

    assert torch.equal(logits, direct_logits)
    assert set(NAMES) <= set(cache.keys())  # a bridge also returns each under its own name for the hook
    assert_same_cache(cache, direct_cache)


def test_the_first_call_writes_one_entry_under_the_models_own_directory(wrapped, tmp_path):
    wrapped.run_with_cache(tokens_of(wrapped), names_filter=NAMES)

    (entry,) = cache_files(tmp_path)
    assert entry.parent.name == wrapped.cfg.model_name  # "custom" for a HookedTransformer built from a config
    assert list(tmp_path.rglob("*.tmp*")) == []  # the write is atomic, nothing half-written is left


def test_a_second_call_is_served_from_disk(wrapped, tmp_path):
    tokens = tokens_of(wrapped)
    logits, cache = wrapped.run_with_cache(tokens, names_filter=NAMES)

    bump_weights(wrapped._model)
    cached_logits, cached_cache = wrapped.run_with_cache(tokens, names_filter=NAMES)
    recomputed_logits, _ = CachedModel(wrapped._model, cache_dir=None).run_with_cache(tokens, names_filter=NAMES)

    # The stored activations answer the second call, and nothing new is written...
    assert torch.equal(cached_logits, logits)
    assert_same_cache(cached_cache, cache)
    assert len(cache_files(tmp_path)) == 1
    # ...while a call that recomputes them follows the changed weights.
    assert not torch.equal(recomputed_logits, logits)


# --------------------------------------------------------------------------------------------------
# Which calls share an entry
# --------------------------------------------------------------------------------------------------
def test_the_order_of_the_names_filter_does_not_matter(wrapped, tmp_path):
    tokens = tokens_of(wrapped)

    _, first = wrapped.run_with_cache(tokens, names_filter=list(reversed(NAMES)))
    _, second = wrapped.run_with_cache(tokens, names_filter=NAMES)

    assert len(cache_files(tmp_path)) == 1
    assert_same_cache(first, second)


def test_a_single_name_is_the_same_request_as_a_one_element_filter(wrapped, tmp_path):
    tokens = tokens_of(wrapped)

    _, from_string = wrapped.run_with_cache(tokens, names_filter=NAMES[0])
    _, from_list = wrapped.run_with_cache(tokens, names_filter=[NAMES[0]])
    _, direct = wrapped._model.run_with_cache(tokens, names_filter=[NAMES[0]])

    assert len(cache_files(tmp_path)) == 1
    assert set(from_string.keys()) == set(from_list.keys()) == set(direct.keys())


def test_a_subset_of_the_names_is_its_own_entry(wrapped, tmp_path):
    tokens = tokens_of(wrapped)

    wrapped.run_with_cache(tokens, names_filter=NAMES[:1])
    wrapped.run_with_cache(tokens, names_filter=NAMES)

    assert len(cache_files(tmp_path)) == 2


def test_an_unfiltered_call_caches_every_activation(wrapped, tmp_path):
    tokens = tokens_of(wrapped)
    _, direct_cache = wrapped._model.run_with_cache(tokens)

    logits, cache = wrapped.run_with_cache(tokens)
    assert set(cache.keys()) == set(direct_cache.keys())
    assert len(cache_files(tmp_path)) == 1

    bump_weights(wrapped._model)
    cached_logits, cached_cache = wrapped.run_with_cache(tokens)
    assert torch.equal(cached_logits, logits)
    assert_same_cache(cached_cache, cache, names=sorted(cache.keys()))
    assert len(cache_files(tmp_path)) == 1


def test_a_prompt_the_model_tokenizes_itself_is_keyed_as_the_string_it_was_given(wrapped, tmp_path):
    logits, cache = wrapped.run_with_cache(PROMPT, names_filter=NAMES)

    bump_weights(wrapped._model)
    cached_logits, cached_cache = wrapped.run_with_cache(PROMPT, names_filter=NAMES)
    assert torch.equal(cached_logits, logits)
    assert_same_cache(cached_cache, cache)
    assert len(cache_files(tmp_path)) == 1


def test_a_prompt_and_its_tokens_are_separate_entries(wrapped, tmp_path):
    # The same forward pass either way, but the key holds the input as it was handed over.
    from_prompt, _ = wrapped.run_with_cache(PROMPT, names_filter=NAMES)
    from_tokens, _ = wrapped.run_with_cache(tokens_of(wrapped), names_filter=NAMES)

    assert torch.equal(from_prompt, from_tokens)
    assert len(cache_files(tmp_path)) == 2


def test_an_input_numpy_cannot_hold_is_keyed_by_its_raw_bytes(backend, tmp_path):
    # A bfloat16 tensor has no NumPy equivalent, so the key reads its bytes through an integer view of
    # the same width. (Only the two-byte side is reachable - no transformer_lens model runs in float8.)
    model = build_small_model(backend, dtype=torch.bfloat16)
    wrapped = CachedModel(model, str(tmp_path))
    residual = model.embed(tokens_of(model))
    assert residual.dtype is torch.bfloat16

    try:
        logits, cache = wrapped.run_with_cache(residual, start_at_layer=0, names_filter=NAMES)
    except NotImplementedError:
        pytest.skip("this transformer-lens's bridge cannot run from a residual")

    bump_weights(model)
    cached_logits, cached_cache = wrapped.run_with_cache(residual, start_at_layer=0, names_filter=NAMES)
    assert torch.equal(cached_logits, logits)
    assert_same_cache(cached_cache, cache)
    assert len(cache_files(tmp_path)) == 1

    # A residual differing in a single value is its own entry: the bytes have to be read, not summarised.
    nudged = residual.clone()
    nudged[0, -1] += 1
    other_logits, _ = wrapped.run_with_cache(nudged, start_at_layer=0, names_filter=NAMES)
    assert len(cache_files(tmp_path)) == 2
    assert not torch.equal(other_logits, cached_logits)


def test_different_tokens_are_different_entries(wrapped, tmp_path):
    first, _ = wrapped.run_with_cache(tokens_of(wrapped, PROMPT), names_filter=NAMES)
    second, _ = wrapped.run_with_cache(tokens_of(wrapped, OTHER_PROMPT), names_filter=NAMES)

    assert len(cache_files(tmp_path)) == 2
    assert not torch.equal(first, second)


def test_two_models_are_kept_apart_by_name(backend, tmp_path):
    # Same shape, same prompt, same activations - only the model differs, and its name becomes the
    # directory (with anything a path cannot carry replaced).
    for name in ("org/tiny-a", "org/tiny-b"):
        model = build_small_model(backend, name=name)
        CachedModel(model, str(tmp_path)).run_with_cache(tokens_of(model), names_filter=NAMES)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["org_tiny-a", "org_tiny-b"]
    assert len(cache_files(tmp_path)) == 2


def test_two_models_are_kept_apart_by_their_weights(backend, tmp_path):
    # A fine-tune reads back the config it was derived from, so the name and every shape match and
    # only the weights tell the two apart. The wrapper takes their fingerprint as it is built, which
    # is why the change here is made before the wrapping and not, as above, after it.
    original, finetuned = build_small_model(backend), build_small_model(backend)
    bump_weights(finetuned)

    for model in (original, finetuned):
        CachedModel(model, str(tmp_path)).run_with_cache(tokens_of(model), names_filter=NAMES)

    assert len(cache_files(tmp_path)) == 2


def test_removing_the_batch_dimension_is_its_own_entry(wrapped, tmp_path):
    tokens = tokens_of(wrapped)
    wrapped.run_with_cache(tokens, names_filter=NAMES)

    _, cache = wrapped.run_with_cache(tokens, names_filter=NAMES, remove_batch_dim=True)
    _, from_disk = wrapped.run_with_cache(tokens, names_filter=NAMES, remove_batch_dim=True)

    assert len(cache_files(tmp_path)) == 2
    for activations in (cache, from_disk):
        assert activations.has_batch_dim is False
        assert activations[NAMES[0]].shape == (tokens.shape[1], wrapped.cfg.d_model)
    assert_same_cache(from_disk, cache)


def test_a_call_that_asks_for_no_logits_round_trips_without_them(wrapped, tmp_path):
    tokens = tokens_of(wrapped)
    wrapped.run_with_cache(tokens, names_filter=NAMES)

    logits, cache = wrapped.run_with_cache(tokens, names_filter=NAMES, return_type=None)
    cached_logits, cached_cache = wrapped.run_with_cache(tokens, names_filter=NAMES, return_type=None)

    assert len(cache_files(tmp_path)) == 2  # the logits are part of what an entry stores
    assert logits is None and cached_logits is None
    assert_same_cache(cached_cache, cache)


def test_a_tensor_valued_model_argument_is_part_of_the_key(wrapped, tmp_path):
    # An attention mask is large enough that ``repr`` would truncate it to the same text either way.
    tokens = tokens_of(wrapped)
    full = torch.ones_like(tokens)
    without_last = full.clone()
    without_last[0, -1] = 0

    attended, _ = wrapped.run_with_cache(tokens, names_filter=NAMES, attention_mask=full)
    masked, _ = wrapped.run_with_cache(tokens, names_filter=NAMES, attention_mask=without_last)

    assert len(cache_files(tmp_path)) == 2
    assert not torch.equal(attended, masked)


# --------------------------------------------------------------------------------------------------
# Requests that fall through uncached
# --------------------------------------------------------------------------------------------------
def test_a_callable_names_filter_is_never_cached(wrapped, tmp_path):
    # A function cannot be keyed, so the call goes straight to the model.
    tokens = tokens_of(wrapped)

    _, cache = wrapped.run_with_cache(tokens, names_filter=lambda name: name.endswith("hook_resid_post"))

    assert set(cache.keys()) == {f"blocks.{layer}.hook_resid_post" for layer in range(wrapped.cfg.n_layers)}
    assert cache_files(tmp_path) == []


def test_a_backward_cache_is_never_cached(wrapped, tmp_path):
    # The gradients come from a backward pass this wrapper does not persist.
    tokens = tokens_of(wrapped)

    loss, cache = wrapped.run_with_cache(tokens, names_filter=NAMES, incl_bwd=True, return_type="loss")
    _, direct = wrapped._model.run_with_cache(tokens, names_filter=NAMES, incl_bwd=True, return_type="loss")

    assert loss.ndim == 0
    # The model's own answer: which names a bridge files the gradients under, if it returns them at all,
    # depends on the transformer-lens release.
    assert set(cache.keys()) == set(direct.keys())
    assert cache_files(tmp_path) == []


# --------------------------------------------------------------------------------------------------
# The wrapper is otherwise the model
# --------------------------------------------------------------------------------------------------
def test_without_a_cache_dir_every_call_recomputes(backend, tmp_path):
    model = build_small_model(backend)
    wrapped = CachedModel(model, cache_dir=None)
    tokens = tokens_of(model)

    logits, cache = wrapped.run_with_cache(tokens, names_filter=NAMES)
    direct_logits, direct_cache = model.run_with_cache(tokens, names_filter=NAMES)
    assert torch.equal(logits, direct_logits)
    assert_same_cache(cache, direct_cache)

    bump_weights(model)
    recomputed, _ = wrapped.run_with_cache(tokens, names_filter=NAMES)
    assert not torch.equal(recomputed, logits)
    assert cache_files(tmp_path) == []


def test_every_other_attribute_is_the_models_own(model, tmp_path):
    wrapped = CachedModel(model, str(tmp_path))

    assert wrapped.cfg is model.cfg
    assert torch.equal(wrapped.W_U, model.W_U)
    assert callable(wrapped.run_with_hooks)
    assert torch.equal(wrapped.to_tokens(PROMPT), model.to_tokens(PROMPT))

    with pytest.raises(AttributeError):
        _ = wrapped.no_such_attribute


def test_a_wrapper_whose_init_never_ran_reports_the_missing_field(model):
    # Without the guard the forwarding would look up the model to find the model: an endless recursion
    # (a RecursionError, which is not what this expects) instead of a plain missing attribute.
    bare = CachedModel.__new__(CachedModel)

    with pytest.raises(AttributeError, match="_model"):
        _ = bare.cfg
