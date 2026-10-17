from typing import Any, TypeAlias

from entoscope.model.cache import CachedModel

try:
    from transformer_lens import HookedTransformer
except ImportError as error:  # transformer-lens 4 removed it in favour of the bridge
    if error.name != "HookedTransformer":  # transformer_lens itself failing to import is a real error
        raise
    HookedTransformer = None

try:
    from transformer_lens.model_bridge import TransformerBridge
except ModuleNotFoundError as error:  # transformer-lens 2.x (Intel macOS fallback)
    if error.name != "transformer_lens.model_bridge":  # anything failing inside it is a real error
        raise
    TransformerBridge = None

if TransformerBridge is None:
    Model: TypeAlias = HookedTransformer | CachedModel
elif HookedTransformer is None:
    Model: TypeAlias = TransformerBridge | CachedModel
else:
    Model: TypeAlias = TransformerBridge | HookedTransformer | CachedModel


def unwrap(model: Model) -> Any:
    """The transformer itself, with any ``CachedModel`` wrapper removed."""
    return getattr(model, "_model", model)


def is_bridge(model: Model) -> bool:
    return TransformerBridge is not None and isinstance(unwrap(model), TransformerBridge)
