from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from transformer_lens import ActivationCache, HookedTransformer

from intex.operation.input import OperationInput
from intex.patching.config import PatchingConfig
from intex.patching.index import PatchingIndex
from intex.patching.metric import LogitDiffMixin


@dataclass
class PatchingInput(OperationInput):
    source_tokens: Int[torch.Tensor, "batch pos"] | None
    target_tokens: Int[torch.Tensor, "batch pos"]
    source_logits: Float[torch.Tensor, "batch pos d_vocab"] | None
    target_logits: Float[torch.Tensor, "batch pos d_vocab"] | None
    source_generation: int | None
    target_generation: int | None
    source_cache: ActivationCache | None
    target_cache: ActivationCache | None
    config: PatchingConfig = None
    index: list[PatchingIndex] | None = None

    _source_baseline: torch.Tensor | None = None
    _target_baseline: torch.Tensor | None = None
    _source_probabilities: torch.Tensor | None = None
    _target_probabilities: torch.Tensor | None = None

    def get_index(self, model: HookedTransformer | None = None) -> list[PatchingIndex]:
        if self.index is None and model is not None:
            index = [PatchingIndex(patching_config, model, self) for patching_config in self.config.patch]
            index_lengths = [len(idx) for idx in index]
            assert all(length == max(index_lengths) or length == 1 for length in index_lengths)
            self.index = index
        assert self.index is not None
        return self.index

    @property
    def source_baseline(self) -> torch.Tensor | None:
        if self._source_baseline is None:
            self._source_baseline = (
                LogitDiffMixin.get_logit_diff(
                    self.source_logits, self.source_logits, self.source_generation, self.target_generation
                )
                if self.source_logits is not None
                and self.source_generation is not None
                and self.target_generation is not None
                else None
            )

        return self._source_baseline

    @property
    def target_baseline(self) -> torch.Tensor | None:
        if self._target_baseline is None:
            self._target_baseline = (
                LogitDiffMixin.get_logit_diff(
                    self.target_logits, self.target_logits, self.source_generation, self.target_generation
                )
                if self.target_logits is not None
                and self.source_generation is not None
                and self.target_generation is not None
                else None
            )

        return self._target_baseline

    @property
    def source_probabilities(self) -> torch.Tensor | None:
        if self._source_probabilities is None:
            self._source_probabilities = self.source_logits.softmax(dim=-1) if self.source_logits is not None else None

        return self._source_probabilities

    @property
    def target_probabilities(self) -> torch.Tensor | None:
        if self._target_probabilities is None:
            self._target_probabilities = self.target_logits.softmax(dim=-1) if self.target_logits is not None else None

        return self._target_probabilities

    def to(self, device: str | torch.device = "cpu") -> "PatchingInput":
        self.source_tokens = self.source_tokens.to(device)
        if self.target_tokens is not None:
            self.target_tokens = self.target_tokens.to(device)
        if self.source_cache is not None:
            self.source_cache = self.source_cache.to(device)
        if self.target_cache is not None:
            self.target_cache = self.target_cache.to(device)
        if self.source_logits is not None:
            self.source_logits = self.source_logits.to(device)
        if self.target_logits is not None:
            self.target_logits = self.target_logits.to(device)

        return self
