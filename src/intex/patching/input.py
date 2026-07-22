from dataclasses import dataclass, field

import torch
from jaxtyping import Float, Int
from transformer_lens import ActivationCache

from intex.model import Model
from intex.model.index import Index
from intex.operation.input import OperationInput
from intex.patching.config import PatchingConfig, PatchingElementConfig
from intex.patching.metric import LogitDiffMixin


@dataclass
class PatchingInput(OperationInput):
    config: PatchingConfig
    source_prompt: str
    target_prompt: str

    _source_tokens: Int[torch.Tensor, "batch pos"] | None = field(init=False, default=None)
    _target_tokens: Int[torch.Tensor, "batch pos"] | None = field(init=False, default=None)
    _source_logits: Float[torch.Tensor, "batch pos d_vocab"] | None = field(init=False, default=None)
    _target_logits: Float[torch.Tensor, "batch pos d_vocab"] | None = field(init=False, default=None)
    _source_generation: int | None = field(init=False, default=None)
    _target_generation: int | None = field(init=False, default=None)
    _source_cache: ActivationCache | None = field(init=False, default=None)
    _target_cache: ActivationCache | None = field(init=False, default=None)

    _index: list[Index[PatchingElementConfig]] | None = field(init=False, default=None)

    _source_baseline: torch.Tensor | None = field(init=False, default=None)
    _target_baseline: torch.Tensor | None = field(init=False, default=None)
    _source_probabilities: torch.Tensor | None = field(init=False, default=None)
    _target_probabilities: torch.Tensor | None = field(init=False, default=None)

    def prepare(self, model: Model) -> "PatchingInput":
        if self._index is None:
            self._source_tokens = model.to_tokens(self.source_prompt)
            self._target_tokens = model.to_tokens(self.target_prompt)

            index = [
                Index[PatchingElementConfig](
                    patching_config.source,
                    patching_config.source.parse(model, self.source_tokens.shape[1]),
                    patching_config.target,
                    patching_config.target.parse(model, self.target_tokens.shape[1]),
                )
                for patching_config in self.config.patch
            ]
            index_lengths = [len(idx) for idx in index]
            assert len(index_lengths) == 0 or all(
                length == max(index_lengths) or length == 1 for length in index_lengths
            )
            self._index = index

            names = sorted(set().union(*(idx.source_activation_names() for idx in index))) if index else []
            with torch.no_grad():
                self._source_logits, self._source_cache = model.run_with_cache(self._source_tokens, names_filter=names)
                self._target_logits, self._target_cache = model.run_with_cache(self._target_tokens, names_filter=names)

            self._source_generation = int(self._source_logits[0, -1].argmax())
            self._target_generation = int(self._target_logits[0, -1].argmax())

        return self

    @property
    def source_tokens(self) -> Int[torch.Tensor, "batch pos"]:
        assert self._source_tokens is not None
        return self._source_tokens

    @property
    def target_tokens(self) -> Int[torch.Tensor, "batch pos"]:
        assert self._target_tokens is not None
        return self._target_tokens

    @property
    def source_logits(self) -> Float[torch.Tensor, "batch pos d_vocab"]:
        assert self._source_logits is not None
        return self._source_logits

    @property
    def target_logits(self) -> Float[torch.Tensor, "batch pos d_vocab"]:
        assert self._target_logits is not None
        return self._target_logits

    @property
    def source_generation(self) -> int:
        assert self._source_generation is not None
        return self._source_generation

    @property
    def target_generation(self) -> int:
        assert self._target_generation is not None
        return self._target_generation

    @property
    def source_cache(self) -> ActivationCache:
        assert self._source_cache is not None
        return self._source_cache

    @property
    def target_cache(self) -> ActivationCache:
        assert self._target_cache is not None
        return self._target_cache

    @property
    def index(self) -> list[Index[PatchingElementConfig]]:
        assert self._index is not None
        return self._index

    @property
    def source_baseline(self) -> torch.Tensor:
        if self._source_baseline is None:
            self._source_baseline = LogitDiffMixin.get_logit_diff(
                self.source_logits, self.source_logits, self.source_generation, self.target_generation
            )
        assert self._source_baseline is not None
        return self._source_baseline

    @property
    def target_baseline(self) -> torch.Tensor:
        if self._target_baseline is None:
            self._target_baseline = LogitDiffMixin.get_logit_diff(
                self.target_logits, self.target_logits, self.source_generation, self.target_generation
            )
        assert self._target_baseline is not None
        return self._target_baseline

    @property
    def source_probabilities(self) -> torch.Tensor:
        if self._source_probabilities is None:
            self._source_probabilities = self.source_logits.softmax(dim=-1)
        assert self._source_probabilities is not None
        return self._source_probabilities

    @property
    def target_probabilities(self) -> torch.Tensor:
        if self._target_probabilities is None:
            self._target_probabilities = self.target_logits.softmax(dim=-1)
        assert self._target_probabilities is not None
        return self._target_probabilities

    def to(self, device: str | torch.device = "cpu") -> "PatchingInput":
        if self._source_tokens is not None:
            self._source_tokens = self._source_tokens.to(device)
        if self._target_tokens is not None:
            self._target_tokens = self._target_tokens.to(device)
        if self._source_cache is not None:
            self._source_cache = self._source_cache.to(device)
        if self._target_cache is not None:
            self._target_cache = self._target_cache.to(device)
        if self._source_logits is not None:
            self._source_logits = self._source_logits.to(device)
        if self._target_logits is not None:
            self._target_logits = self._target_logits.to(device)
        if self._source_baseline is not None:
            self._source_baseline = self._source_baseline.to(device)
        if self._target_baseline is not None:
            self._target_baseline = self._target_baseline.to(device)
        if self._source_probabilities is not None:
            self._source_probabilities = self._source_probabilities.to(device)
        if self._target_probabilities is not None:
            self._target_probabilities = self._target_probabilities.to(device)

        return self
