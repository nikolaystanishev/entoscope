from dataclasses import dataclass, field

import torch
from jaxtyping import Int
from transformer_lens import ActivationCache

from entoscope.lens.config import LensConfig
from entoscope.lens.method.factory import LensMethodFactory
from entoscope.model import Model
from entoscope.model.config import ModelLocatorConfig
from entoscope.model.index import Index
from entoscope.operation.input import OperationInput


@dataclass
class LensInput(OperationInput):
    """A lens config and the prompt it reads.

    Attributes:
        config: The lens config.
        prompt: The prompt whose activations the lens reads, ``None`` exactly when the lens reads the
            model's weights instead, as ``parameter_lens`` does.

    Raises:
        ValueError: If the prompt is missing for a lens that reads activations, or given to one that
            reads weights.
    """

    config: LensConfig
    prompt: str | None = None

    _tokens: Int[torch.Tensor, "batch pos"] | None = field(init=False, default=None)
    _cache: ActivationCache | None = field(init=False, default=None)

    _index: Index[ModelLocatorConfig] | None = field(init=False, default=None)

    def __post_init__(self):
        name = self.config.method.type
        reads_activations = LensMethodFactory.get(name).reads_activations
        if reads_activations and self.prompt is None:
            raise ValueError(f"A '{name}' reads activations, so it needs a prompt")
        if not reads_activations and self.prompt is not None:
            raise ValueError(f"A '{name}' reads the model's weights, not its activations, so it takes no prompt")

    def prepare(self, model: Model, *, index_only: bool = False) -> "LensInput":
        if self._index is None:
            n_positions = None
            if self.prompt is not None:
                self._tokens = model.to_tokens(self.prompt)
                n_positions = self.tokens.shape[1]
            self._index = Index[ModelLocatorConfig](self.config.source, self.config.source.parse(model, n_positions))

        if not index_only and self.prompt is not None and self._cache is None:
            with torch.no_grad():
                _, self._cache = model.run_with_cache(
                    self._tokens, names_filter=sorted(self.index.source_activation_names())
                )

        return self

    @property
    def tokens(self) -> Int[torch.Tensor, "batch pos"]:
        """The prompt's tokens, once the index is prepared."""
        assert self._tokens is not None
        return self._tokens

    @property
    def cache(self) -> ActivationCache:
        """The activations the index reads, once the input is prepared."""
        assert self._cache is not None
        return self._cache

    @property
    def index(self) -> Index[ModelLocatorConfig]:
        """The rows the lens reads, once the index is prepared."""
        assert self._index is not None
        return self._index

    def to(self, device: str | torch.device = "cpu") -> "LensInput":
        if self._tokens is not None:
            self._tokens = self._tokens.to(device)
        if self._cache is not None:
            self._cache = self._cache.to(device)

        return self
