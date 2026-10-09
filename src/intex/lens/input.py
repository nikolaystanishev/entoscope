from dataclasses import dataclass, field

import torch
from jaxtyping import Float, Int
from transformer_lens import ActivationCache

from intex.lens.config import LensConfig
from intex.lens.method.factory import LensMethodFactory
from intex.model import Model
from intex.model.config import ModelLocation, ModelLocatorConfig
from intex.model.index import Index
from intex.operation.input import OperationInput


@dataclass
class LensInput(OperationInput):
    config: LensConfig
    prompt: str | None = None

    _tokens: Int[torch.Tensor, "batch pos"] | None = field(init=False, default=None)
    _logits: Float[torch.Tensor, "batch pos d_vocab"] | None = field(init=False, default=None)
    _cache: ActivationCache | None = field(init=False, default=None)

    _index: Index[ModelLocatorConfig] | None = field(init=False, default=None)

    def __post_init__(self):
        name = self.config.method
        if LensMethodFactory.get(name).reads_activations:
            if self.prompt is None:
                raise ValueError(f"A '{name}' reads activations, so it needs a prompt")
        else:
            if self.prompt is not None:
                raise ValueError(f"A '{name}' reads the model's weights, not its activations, so it takes no prompt")
            if ModelLocation.POSITION in self.config.source.location or any(
                ModelLocation.POSITION in coordinate for coordinate in self.config.source.coordinates or []
            ):
                raise ValueError(
                    f"'position' is not a location of '{name}': it reads the model's weights, "
                    "which have no position axis"
                )

    def prepare_index(self, model: Model) -> "LensInput":
        if self._index is not None:
            return self

        if self.prompt is None:
            self._index = Index[ModelLocatorConfig](self.config.source, self.config.source.parse(model, None))
            return self

        self._tokens = model.to_tokens(self.prompt)
        self._index = Index[ModelLocatorConfig](
            self.config.source, self.config.source.parse(model, self.tokens.shape[1])
        )

        return self

    def prepare(self, model: Model) -> "LensInput":
        self.prepare_index(model)

        # Guarded on the cache, not the index: a ``plan`` may have resolved the index already.
        if self.prompt is not None and self._cache is None:
            with torch.no_grad():
                self._logits, self._cache = model.run_with_cache(
                    self._tokens, names_filter=sorted(self.index.source_activation_names())
                )

        return self

    @property
    def tokens(self) -> Int[torch.Tensor, "batch pos"]:
        assert self._tokens is not None
        return self._tokens

    @property
    def logits(self) -> Float[torch.Tensor, "batch pos d_vocab"]:
        assert self._logits is not None
        return self._logits

    @property
    def cache(self) -> ActivationCache:
        assert self._cache is not None
        return self._cache

    @property
    def index(self) -> Index[ModelLocatorConfig]:
        assert self._index is not None
        return self._index

    def to(self, device: str | torch.device = "cpu") -> "LensInput":
        if self._tokens is not None:
            self._tokens = self._tokens.to(device)
        if self._logits is not None:
            self._logits = self._logits.to(device)
        if self._cache is not None:
            self._cache = self._cache.to(device)

        return self
