from typing import TypeAlias

from transformer_lens import HookedTransformer

from intex.model.cache import CachedModel

# Anything operations accept in place of a raw model: the transformer itself, or the transparent
# disk-caching wrapper around it. CachedModel forwards every attribute to the wrapped model via
# __getattr__, so it is a runtime drop-in; this alias makes it a static drop-in too.
Model: TypeAlias = HookedTransformer | CachedModel
