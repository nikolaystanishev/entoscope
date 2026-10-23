"""Local-storage caching of model activations.

``CachedModel`` is a transparent wrapper around a ``transformer_lens`` model, a
``TransformerBridge`` or a ``HookedTransformer``, that memorizes ``run_with_cache`` to disk.
Operations receive it in place of the raw model and never know the difference: every attribute
other than ``run_with_cache`` is forwarded to the wrapped model, so tokenization,
``run_with_hooks`` (patching), weight access and ``cfg`` behave exactly as before, while repeated
forward passes for the same ``(model, tokens, names_filter)`` are read back from ``cache_dir``
instead of recomputed. A forwarded method that hands back the model itself, as ``to`` and ``eval``
do, hands back the wrapper, so ``model = model.to(device)`` keeps the cache.

Usage::

    model = TransformerBridge.boot_transformers(config.model_name)
    model = CachedModel(model, cache_dir="./.activation_cache")  # cache_dir=None => pass-through
    Lens(model).apply(LensInput(LensConfig(...), "Paris is located in"))
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import os
import re
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from transformer_lens import ActivationCache

from entoscope.util.data import create_if_not_exists

if TYPE_CHECKING:
    from entoscope.model.base import Model

FORMAT_VERSION = 1

_NON_DIGEST_KWARGS = frozenset({"names_filter", "remove_batch_dim", "incl_bwd", "device"})

_ALL_ACTIVATIONS = "*ALL*"

_WEIGHT_SAMPLE = 64

# torch dtypes ``Tensor.numpy()`` cannot materialize; their raw bytes are reinterpreted instead.
_NUMPY_UNSUPPORTED_DTYPES = frozenset(
    dtype
    for name in ("bfloat16", "float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")
    if (dtype := getattr(torch, name, None)) is not None
)


def _normalize_names_filter(names_filter: Any) -> str | list[str]:
    """Put a ``names_filter`` into a canonical form that does not depend on the order of its names.

    Callables are not supported and are handled by the caller.

    Args:
        names_filter: ``None`` for every activation, a transformer activation name, or an iterable of them.

    Returns:
        A sentinel string for ``None``, otherwise the names as a sorted list.
    """
    if names_filter is None:
        return _ALL_ACTIVATIONS
    if isinstance(names_filter, str):
        return [names_filter]
    return sorted(str(name) for name in names_filter)


def _encode_input(model_input: Any) -> bytes:
    """Encode one ``run_with_cache`` argument as bytes, the same in every process.

    Args:
        model_input: A tensor, a string, or anything with a stable ``repr``.

    Returns:
        The encoding, tagged with the kind of value it encodes.
    """
    if isinstance(model_input, torch.Tensor):
        tensor = model_input.detach().cpu().contiguous()
        header = f"{tensor.dtype}|{tuple(tensor.shape)}|".encode()
        if tensor.dtype in _NUMPY_UNSUPPORTED_DTYPES:
            tensor = tensor.view(torch.int16 if tensor.element_size() == 2 else torch.uint8)
        return b"T|" + header + tensor.numpy().tobytes()
    if isinstance(model_input, str):
        return b"S|" + model_input.encode("utf-8")
    return b"R|" + repr(model_input).encode("utf-8")


def _weight_digest(model: Any) -> str:
    """Digest the weights a model holds, not the architecture its ``cfg`` describes.

    Two models can read back the same ``cfg`` and the same name and still compute different things (a
    fine-tune, a merged adapter, a supplied ``hf_model``, a checkpoint at another revision), and
    without this they would share cache entries. Hashing every parameter in full would cost more than
    the forward pass the cache saves, so each one contributes its name, its layout and the first
    ``_WEIGHT_SAMPLE`` of its values: bounded by the parameter count rather than the parameter size,
    and deterministic, so an entry written by one process is still found by the next. It is taken on
    every call, so the samples leave the device in one transfer per device and dtype rather than one
    per parameter.

    Args:
        model: The transformer.

    Returns:
        A SHA-256 hex digest.
    """
    digest = hashlib.sha256()
    samples: defaultdict[tuple[str, str], list[torch.Tensor]] = defaultdict(list)
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        digest.update(f"{name}|{tuple(parameter.shape)}|{parameter.dtype}|".encode())
        samples[(str(parameter.device), str(parameter.dtype))].append(parameter.detach().flatten()[:_WEIGHT_SAMPLE])
    for group in sorted(samples):
        digest.update(_encode_input(torch.cat(samples[group])))
    return digest.hexdigest()


def _compact(tensor: torch.Tensor) -> torch.Tensor:
    """Store a tensor whose batch rows are all alike as one row, expanded back to its shape.

    ``torch.save`` writes a view's storage rather than its shape, so the entry holds the row once and
    ``torch.load`` hands back the expanded view. Patching runs each prompt as wide as its patched runs,
    which is where such rows come from. Rows are compared exactly, so on a backend that rounds the copies
    of one prompt apart, as some x86 CPUs do, every row is kept, and an entry always reads back exactly
    what the model computed.

    Args:
        tensor: A cached tensor, its batch rows along the first axis.

    Returns:
        The tensor, or the view of its single row when every row equals the first.
    """
    if tensor.ndim == 0 or tensor.shape[0] < 2:
        return tensor
    row = tensor[:1].clone()
    return row.expand_as(tensor) if torch.equal(tensor, row.expand_as(tensor)) else tensor


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


class ActivationDiskCache:
    """Content-addressed on-disk store for ``(logits, activation cache)`` pairs.

    Pure filesystem and key logic with no dependency on a live model, so the key derivation can be
    tested directly. Entries live at ``<cache_dir>/<model_name>/<digest>.pt`` and are written
    atomically, through a temporary file and ``os.replace``, so an interrupted run never leaves a
    half-written file.

    Args:
        cache_dir: The directory entries are stored under.
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
        """Derive the key of one forward pass.

        Args:
            model_fingerprint: What identifies the model, its architecture and weights.
            model_args: The positional arguments of the ``run_with_cache`` call.
            names_filter: The activations cached, as ``_normalize_names_filter`` returns them.
            extra: Any other keyword argument that changes the result.

        Returns:
            A SHA-256 hex digest.
        """
        digest = hashlib.sha256()
        # An entry written in another layout lives under another key, so it is never read back.
        digest.update(f"format {FORMAT_VERSION}\x00".encode())
        digest.update(model_fingerprint.encode("utf-8"))
        for arg in model_args:
            digest.update(b"\x00")
            digest.update(_encode_input(arg))
        digest.update(b"\x00names\x00")
        names = names_filter if isinstance(names_filter, str) else "|".join(names_filter)
        digest.update(names.encode("utf-8"))
        digest.update(b"\x00extra\x00")
        for extra_name, extra_value in sorted(extra.items()):
            digest.update(extra_name.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(_encode_input(extra_value))
            digest.update(b"\x00")
        return digest.hexdigest()

    def _path(self, digest: str, model_name: str) -> str:
        return os.path.join(self.cache_dir, _sanitize(model_name), f"{digest}.pt")

    def load(self, digest: str, model_name: str) -> dict | None:
        """Read an entry back.

        Args:
            digest: The entry's key.
            model_name: The model the entry belongs to.

        Returns:
            The stored payload, on the CPU, or ``None`` when there is no such entry or it was written in
            another format.
        """
        path = self._path(digest, model_name)
        if not os.path.exists(path):
            return None
        payload = torch.load(path, map_location="cpu")
        return payload if payload.get("format_version") == FORMAT_VERSION else None

    def store(self, digest: str, model_name: str, payload: dict) -> None:
        """Write an entry atomically.

        Args:
            digest: The entry's key.
            model_name: The model the entry belongs to.
            payload: What to store.
        """
        path = self._path(digest, model_name)
        create_if_not_exists(os.path.dirname(path))
        tmp_path = f"{path}.tmp.{os.getpid()}"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)


class CachedModel:
    """Transparent read-through disk cache around a ``transformer_lens`` model.

    Overrides only ``run_with_cache``, and forwards every other attribute to the wrapped model, a method
    that returns the model itself returning the wrapper instead. Cache keys fold in a sampled digest of
    the model's weights, taken on every call, so neither a fine-tune nor an edit made after wrapping,
    such as ``enable_compatibility_mode()`` or a change of dtype, reads the entries of the weights it
    replaced.

    Args:
        model: A ``TransformerBridge`` or a ``HookedTransformer``.
        cache_dir: The directory activations are stored under. ``None`` makes the wrapper a pure
            pass-through.
    """

    def __init__(self, model: Model, cache_dir: str | None = None):
        self._model = model
        self._cache = ActivationDiskCache(cache_dir) if cache_dir else None

    def __getattr__(self, name: str) -> Any:
        if name in {"_model", "_cache"}:
            raise AttributeError(name)
        attribute = getattr(self._model, name)
        if inspect.ismethod(attribute) and attribute.__self__ is self._model:
            return self._keeping_the_wrapper(attribute)
        return attribute

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)

    def _keeping_the_wrapper(self, method: Callable) -> Callable:
        """Forward a method of the model, handing back the wrapper wherever the method hands back the model.

        Args:
            method: A method bound to the wrapped model.

        Returns:
            The method, returning the wrapper in place of the model.
        """

        @functools.wraps(method)
        def forwarded(*args: Any, **kwargs: Any) -> Any:
            result = method(*args, **kwargs)
            return self if result is self._model else result

        return forwarded

    def _fingerprint(self) -> str:
        cfg = self._model.cfg
        parts = (
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
            _weight_digest(self._model),
        )
        return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()

    def run_with_cache(self, *model_args: Any, **kwargs: Any) -> tuple[Any, ActivationCache]:
        """Run the model and cache its activations, reading an earlier identical run back from disk.

        A callable ``names_filter`` or ``incl_bwd=True`` cannot be keyed, so such a call always runs
        the model.

        Args:
            *model_args: The wrapped model's ``run_with_cache`` positional arguments, such as tokens.
            **kwargs: Its keyword arguments, such as ``names_filter``.

        Returns:
            The logits and the activation cache, on the model's device.
        """
        names_filter = kwargs.get("names_filter")

        if self._cache is None or isinstance(names_filter, Callable) or kwargs.get("incl_bwd"):
            return self._model.run_with_cache(*model_args, **kwargs)

        model_name = self._model.cfg.model_name or "unknown"
        names_normalized = _normalize_names_filter(names_filter)
        remove_batch_dim = bool(kwargs.get("remove_batch_dim", False))
        extra = {key: value for key, value in kwargs.items() if key not in _NON_DIGEST_KWARGS}
        extra["remove_batch_dim"] = remove_batch_dim

        fingerprint = self._fingerprint()
        digest = self._cache.key(
            model_fingerprint=fingerprint,
            model_args=model_args,
            names_filter=names_normalized,
            extra=extra,
        )

        payload = self._cache.load(digest, model_name)
        if payload is not None:
            return self._rebuild(payload)

        logits, cache = self._model.run_with_cache(*model_args, **kwargs)

        def stored(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor.detach().cpu()
            return tensor if remove_batch_dim else _compact(tensor)

        self._cache.store(
            digest,
            model_name,
            {
                "format_version": FORMAT_VERSION,
                "model_name": model_name,
                "model_fingerprint": fingerprint,
                "names_filter": names_normalized,
                "remove_batch_dim": remove_batch_dim,
                "logits": None if logits is None else stored(logits),
                "cache": {name: stored(tensor) for name, tensor in cache.cache_dict.items()},
            },
        )
        return logits, cache

    def _rebuild(self, payload: dict) -> tuple[Any, ActivationCache]:
        device = self._model.cfg.device
        logits = payload["logits"]
        # ``contiguous`` materializes a row stored once, so the tensors can be written to as computed ones can.
        if logits is not None:
            logits = logits.to(device).contiguous()
        cache_dict = {name: tensor.to(device).contiguous() for name, tensor in payload["cache"].items()}
        cache = ActivationCache(cache_dict, self._model, has_batch_dim=not payload["remove_batch_dim"])
        return logits, cache
