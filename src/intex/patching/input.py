from dataclasses import dataclass, field

import torch
from jaxtyping import Float, Int
from transformer_lens import ActivationCache

from intex.model import Model
from intex.model.index import Index
from intex.operation.input import OperationInput
from intex.patching.config import PatchingConfig, PatchingElementConfig, PatchingMetric
from intex.patching.metric import AbstractComparison, AbstractMetric


@dataclass
class PatchingInput(OperationInput):
    config: PatchingConfig
    source_prompt: str | None
    target_prompt: str
    correct_token: str | int | None = None
    distractor_token: str | int | None = None

    _source_tokens: Int[torch.Tensor, "batch pos"] | None = field(init=False, default=None)
    _target_tokens: Int[torch.Tensor, "batch pos"] | None = field(init=False, default=None)
    _source_logits: Float[torch.Tensor, "batch pos d_vocab"] | None = field(init=False, default=None)
    _target_logits: Float[torch.Tensor, "batch pos d_vocab"] | None = field(init=False, default=None)
    _source_generation: int | None = field(init=False, default=None)
    _target_generation: int | None = field(init=False, default=None)
    _correct_token_id: int | None = field(init=False, default=None)
    _distractor_token_id: int | None = field(init=False, default=None)
    _source_cache: ActivationCache | None = field(init=False, default=None)
    _target_cache: ActivationCache | None = field(init=False, default=None)

    _index: list[Index[PatchingElementConfig]] | None = field(init=False, default=None)

    _baselines: dict[tuple[str, PatchingMetric], torch.Tensor] = field(init=False, default_factory=dict)
    _source_probabilities: torch.Tensor | None = field(init=False, default=None)
    _target_probabilities: torch.Tensor | None = field(init=False, default=None)

    def prepare_index(self, model: Model) -> "PatchingInput":
        if self._index is not None:
            return self

        if self.source_prompt is not None:
            self._source_tokens = model.to_tokens(self.source_prompt)
        self._target_tokens = model.to_tokens(self.target_prompt)

        index = [
            Index[PatchingElementConfig](
                patching_config.source,
                patching_config.source.parse(
                    model, self.source_tokens.shape[1] if self._source_tokens is not None else None
                ),
                patching_config.target,
                patching_config.target.parse(model, self.target_tokens.shape[1]),
            )
            for patching_config in self.config.patch
        ]
        index_lengths = [len(idx) for idx in index]
        assert len(index_lengths) == 0 or all(length == max(index_lengths) or length == 1 for length in index_lengths)
        self._index = index

        return self

    def prepare(self, model: Model) -> "PatchingInput":
        self.prepare_index(model)
        self._check_metrics_are_available()

        # Guarded on the target cache, not the index: a ``plan`` may have resolved the index
        # already, and a run without a source prompt never fills the source one.
        if self._target_cache is None:
            names = self.activation_names()
            with torch.no_grad():
                if self._source_tokens is not None:
                    self._source_logits, self._source_cache = model.run_with_cache(
                        self._source_tokens, names_filter=names
                    )
                self._target_logits, self._target_cache = model.run_with_cache(self._target_tokens, names_filter=names)

            if self._source_logits is not None:
                self._source_generation = int(self._source_logits[0, -1].argmax())
            self._target_generation = int(self._target_logits[0, -1].argmax())
            self._resolve_answer_tokens(model)

        return self

    def activation_names(self) -> list[str]:
        """Every activation the run touches, on either side of the patch.

        The target's own names are read only by the gradient methods, which need the clean
        activation a patch would overwrite and the value they interpolate towards; the two sides
        name the same activation in the common case, so caching both usually costs nothing.
        """
        names: set[str] = set()
        for idx in self.index:
            names |= idx.source_activation_names() | idx.target_activation_names()
        return sorted(names)

    def target_activation_names(self) -> list[str]:
        return sorted(set().union(*(idx.target_activation_names() for idx in self.index))) if self.index else []

    def _check_metrics_are_available(self) -> None:
        """Name every metric and comparison, then reject those that need a source run there is not.

        Resolved up front rather than at the first cell, so a config naming something no metric
        implements fails before the model is run. Both are collected into dicts, so a comparison
        named under several metrics is reported once.
        """
        metrics = {metric: AbstractMetric.get(metric) for metric in self.config.metric}
        comparisons = {
            comparison: AbstractComparison.get(comparison)
            for comparisons in self.config.metric.values()
            for comparison in comparisons
        }
        if self.source_prompt is not None:
            return

        unavailable = [
            named.value
            for named, implementation in (*metrics.items(), *comparisons.items())
            if implementation.needs_source_run
        ]
        if unavailable:
            raise ValueError(
                f"{', '.join(unavailable)} scores the patched run against the source run, which needs a source prompt"
            )

    def _resolve_answer_tokens(self, model: Model) -> None:
        self._correct_token_id = self._token_id(model, self.correct_token, self._source_generation)
        self._distractor_token_id = self._token_id(model, self.distractor_token, self._target_generation)

        if self._correct_token_id == self._distractor_token_id and any(
            AbstractMetric.get(metric).needs_distractor for metric in self.config.metric
        ):
            token = model.to_string([self._correct_token_id])
            raise ValueError(
                f"The correct and distractor tokens are both {self._correct_token_id} ({token!r}), so every "
                "difference metric is identically zero. Name them with 'correct_token' and 'distractor_token'."
            )

    @staticmethod
    def _token_id(model: Model, token: str | int | None, fallback: int | None) -> int:
        if isinstance(token, str):
            return model.to_single_token(token)
        if token is not None:
            return int(token)
        if fallback is None:
            raise ValueError(
                "Without a source prompt there is no run to read the correct token off — pass 'correct_token'"
            )
        return fallback

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
    def correct_token_id(self) -> int:
        assert self._correct_token_id is not None
        return self._correct_token_id

    @property
    def distractor_token_id(self) -> int:
        assert self._distractor_token_id is not None
        return self._distractor_token_id

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

    def source_baseline(self, metric: PatchingMetric) -> torch.Tensor:
        return self._baseline("source", metric)

    def target_baseline(self, metric: PatchingMetric) -> torch.Tensor:
        return self._baseline("target", metric)

    def _baseline(self, run: str, metric: PatchingMetric) -> torch.Tensor:
        """``metric`` on one of the two unpatched runs — what a patched run is reported against."""
        if (run, metric) not in self._baselines:
            logits = self.source_logits if run == "source" else self.target_logits
            self._baselines[(run, metric)] = AbstractMetric.get(metric).compute(self, logits[0, -1])
        return self._baselines[(run, metric)]

    @property
    def source_probabilities(self) -> torch.Tensor:
        if self._source_probabilities is None:
            self._source_probabilities = self.source_logits[0, -1].softmax(dim=-1)
        return self._source_probabilities

    @property
    def target_probabilities(self) -> torch.Tensor:
        if self._target_probabilities is None:
            self._target_probabilities = self.target_logits[0, -1].softmax(dim=-1)
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
        if self._source_probabilities is not None:
            self._source_probabilities = self._source_probabilities.to(device)
        if self._target_probabilities is not None:
            self._target_probabilities = self._target_probabilities.to(device)
        self._baselines = {key: baseline.to(device) for key, baseline in self._baselines.items()}

        return self
