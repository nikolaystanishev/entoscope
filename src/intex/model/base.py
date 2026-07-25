from typing import Any, TypeAlias

from transformer_lens import HookedTransformer

from intex.model.cache import CachedModel

try:
    from transformer_lens.model_bridge import TransformerBridge
except ImportError:  # transformer-lens 2.x (Intel macOS fallback)
    TransformerBridge = None

# Anything operations accept in place of a raw model: a transformer, or the transparent
# disk-caching wrapper around one. CachedModel forwards every attribute to the wrapped model via
# __getattr__, so it is a runtime drop-in; this alias makes it a static drop-in too.
if TransformerBridge is None:
    Model: TypeAlias = HookedTransformer | CachedModel
else:
    Model: TypeAlias = TransformerBridge | HookedTransformer | CachedModel


def unwrap(model: Model) -> Any:
    """The transformer itself, with any ``CachedModel`` wrapper removed."""
    return getattr(model, "_model", model)


def is_bridge(model: Model) -> bool:
    return TransformerBridge is not None and isinstance(unwrap(model), TransformerBridge)
