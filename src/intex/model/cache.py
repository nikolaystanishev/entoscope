"""Local-storage caching of model activations.

``CachedModel`` is a transparent wrapper around a ``transformer_lens`` model — a
``TransformerBridge`` or a ``HookedTransformer`` — that memorizes ``run_with_cache`` to disk.
Operations receive it in place of the raw model and never know the difference: every attribute
other than ``run_with_cache`` is forwarded to the wrapped model, so tokenization,
``run_with_hooks`` (patching), weight access and ``cfg`` behave exactly as before, while repeated
forward passes for the same ``(model, tokens, names_filter)`` are read back from ``cache_dir``
instead of recomputed.

Usage::

    model = TransformerBridge.boot_transformers(config.model_name)
    model = CachedModel(model, cache_dir="./.activation_cache")  # cache_dir=None => pass-through
    Lens(model).apply(LensInput(LensConfig(...), "Paris is located in"))
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from transformer_lens import ActivationCache

from intex.util.data import create_if_not_exists

if TYPE_CHECKING:  # ``intex.model.base`` imports this module, so the alias is type-only.
    from intex.model.base import Model

FORMAT_VERSION = 1

# ``run_with_cache`` kwargs that must NOT be folded into the cache key here: ``names_filter`` and
# ``remove_batch_dim`` are keyed explicitly, ``incl_bwd`` diverts uncacheable requests upstream, and
# ``device`` only moves the result. Every other kwarg (return_type, prepend_bos, attention_mask,
# tokens, ...) changes the returned logits/cache and so is folded into the key verbatim.
_NON_DIGEST_KWARGS = frozenset({"names_filter", "remove_batch_dim", "incl_bwd", "device"})

_ALL_ACTIVATIONS = "*ALL*"

# torch dtypes ``Tensor.numpy()`` cannot materialize; their raw bytes are reinterpreted instead.
_NUMPY_UNSUPPORTED_DTYPES = frozenset(
    dtype
    for name in ("bfloat16", "float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")
    if (dtype := getattr(torch, name, None)) is not None
)


def _normalize_names_filter(names_filter: Any) -> str | list[str]:
    """Canonical, hashable form of a ``names_filter`` (order-independent).

    ``None`` (capture everything) becomes a sentinel; a single name becomes a one-element list;
    any iterable of names becomes a sorted list. Callables are unsupported and rejected upstream.
    """
    if names_filter is None:
        return _ALL_ACTIVATIONS
    if isinstance(names_filter, str):
        return [names_filter]
    return sorted(str(name) for name in names_filter)


def _encode_input(model_input: Any) -> bytes:
    """Deterministic byte encoding of one ``run_with_cache`` positional argument."""
    if isinstance(model_input, torch.Tensor):
        tensor = model_input.detach().cpu().contiguous()
        header = f"{tensor.dtype}|{tuple(tensor.shape)}|".encode()
        # ``Tensor.numpy()`` rejects dtypes NumPy has no equivalent for (bfloat16, float8, ...).
        # Reinterpret their raw bytes through a same-width integer view; the original dtype is
        # already recorded in ``header`` so distinct dtypes can never collide on the same bytes.
        if tensor.dtype in _NUMPY_UNSUPPORTED_DTYPES:
            tensor = tensor.view(torch.int16 if tensor.element_size() == 2 else torch.uint8)
        return b"T|" + header + tensor.numpy().tobytes()
    if isinstance(model_input, str):
        return b"S|" + model_input.encode("utf-8")
    return b"R|" + repr(model_input).encode("utf-8")


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


class ActivationDiskCache:
    """Content-addressed on-disk store for ``(logits, activation cache)`` pairs.

    Pure filesystem/key logic with no dependency on a live model, so the key derivation can be
    unit-tested directly. Entries live at ``<cache_dir>/<model_name>/<digest>.pt`` and are written
    atomically (temp file + ``os.replace``) so an interrupted run never leaves a half-written file.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir

    def key(
        self,
        *,
        model_fingerprint: str,
        model_args: tuple[Any, ...],
        names_filter: str | list[str],
        extra: dict[str, Any],
    ) -> str:
        digest = hashlib.sha256()
        digest.update(model_fingerprint.encode("utf-8"))
        for arg in model_args:
            digest.update(b"\x00")
            digest.update(_encode_input(arg))
        digest.update(b"\x00names\x00")
        names = names_filter if isinstance(names_filter, str) else "|".join(names_filter)
        digest.update(names.encode("utf-8"))
        digest.update(b"\x00extra\x00")
        # Values may be tensors (e.g. ``attention_mask``), so reuse the deterministic input encoder
        # rather than ``repr`` — ``repr`` truncates large tensors and would let distinct inputs collide.
        for extra_name, extra_value in sorted(extra.items()):
            digest.update(extra_name.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(_encode_input(extra_value))
            digest.update(b"\x00")
        return digest.hexdigest()

    def _path(self, digest: str, model_name: str) -> str:
        return os.path.join(self.cache_dir, _sanitize(model_name), f"{digest}.pt")

    def load(self, digest: str, model_name: str) -> dict | None:
        path = self._path(digest, model_name)
        if not os.path.exists(path):
            return None
        return torch.load(path, map_location="cpu")

    def store(self, digest: str, model_name: str, payload: dict) -> None:
        path = self._path(digest, model_name)
        create_if_not_exists(os.path.dirname(path))
        tmp_path = f"{path}.tmp.{os.getpid()}"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)


class CachedModel:
    """Transparent read-through disk cache around a ``transformer_lens`` model.

    Overrides only ``run_with_cache``; every other attribute is forwarded to the wrapped model via
    ``__getattr__``. When ``cache_dir`` is ``None`` the wrapper is a pure pass-through.
    """

    def __init__(self, model: Model, cache_dir: str | None = None):
        self._model = model
        self._cache = ActivationDiskCache(cache_dir) if cache_dir else None

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not found on the wrapper itself. Guard the private fields so a
        # partially-initialized or copied instance raises AttributeError instead of recursing.
        if name in {"_model", "_cache"}:
            raise AttributeError(name)
        return getattr(self._model, name)

    def _fingerprint(self) -> str:
        cfg = self._model.cfg
        parts = (
            # A bridge and a HookedTransformer of the same name read the same cfg but not
            # necessarily the same weights, so they must not share cache entries.
            type(self._model).__name__,
            cfg.model_name,
            cfg.n_layers,
            cfg.n_heads,
            cfg.n_key_value_heads,
            cfg.d_model,
            cfg.d_head,
            cfg.d_mlp,
            cfg.d_vocab,
            str(cfg.dtype),
            cfg.act_fn,
        )
        return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()

    def run_with_cache(self, *model_args: Any, **kwargs: Any) -> tuple[Any, ActivationCache]:
        names_filter = kwargs.get("names_filter")

        # Uncacheable requests fall straight through: a callable filter cannot be keyed, and a
        # backward cache would need a backward pass we do not persist. intex uses neither.
        if self._cache is None or isinstance(names_filter, Callable) or kwargs.get("incl_bwd"):
            return self._model.run_with_cache(*model_args, **kwargs)

        model_name = self._model.cfg.model_name or "unknown"
        names_normalized = _normalize_names_filter(names_filter)
        remove_batch_dim = bool(kwargs.get("remove_batch_dim", False))
        extra = {key: value for key, value in kwargs.items() if key not in _NON_DIGEST_KWARGS}
        extra["remove_batch_dim"] = remove_batch_dim

        digest = self._cache.key(
            model_fingerprint=self._fingerprint(),
            model_args=model_args,
            names_filter=names_normalized,
            extra=extra,
        )

        payload = self._cache.load(digest, model_name)
        if payload is not None:
            return self._rebuild(payload)

        logits, cache = self._model.run_with_cache(*model_args, **kwargs)
        self._cache.store(
            digest,
            model_name,
            {
                "format_version": FORMAT_VERSION,
                "model_name": model_name,
                "model_fingerprint": self._fingerprint(),
                "names_filter": names_normalized,
                "remove_batch_dim": remove_batch_dim,
                "logits": None if logits is None else logits.detach().cpu(),
                "cache": {name: tensor.detach().cpu() for name, tensor in cache.cache_dict.items()},
            },
        )
        return logits, cache

    def _rebuild(self, payload: dict) -> tuple[Any, ActivationCache]:
        device = self._model.cfg.device
        logits = payload["logits"]
        if logits is not None:
            logits = logits.to(device)
        cache_dict = {name: tensor.to(device) for name, tensor in payload["cache"].items()}
        cache = ActivationCache(cache_dict, self._model, has_batch_dim=not payload["remove_batch_dim"])
        return logits, cache
