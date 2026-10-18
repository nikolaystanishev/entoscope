from dataclasses import dataclass, field

import torch
from jaxtyping import Float, Int
from transformer_lens import ActivationCache

from entoscope.model import Model
from entoscope.model.index import Index
from entoscope.operation.input import OperationInput
from entoscope.patching.config import PatchingConfig, PatchingElementConfig, PatchingMetric
from entoscope.patching.metric import AbstractComparison, AbstractMetric
from entoscope.patching.processor import SourceStrategy, TargetStrategy
from entoscope.util.collections import broadcast_length, zip_broadcast


@dataclass
class PatchingInput(OperationInput):
    """The two prompts a patching run sits between, and the tokens its metrics compare.

    The target run is the one patched, and the source run is what ``replace`` and ``value_scale`` read.
    A left-unset answer token only means something while the two prompts disagree, so a pair that
    predicts the same token is rejected by the difference metrics.

    Attributes:
        config: The patching config.
        source_prompt: The prompt patched from. ``None`` when no element reads it, and then
            ``kl_divergence`` and ``normalized``, which score against the source run, are unavailable.
        target_prompt: The prompt patched into.
        correct_token: The ``r`` of the metrics, a string that tokenizes to a single token or a token id.
            Unset, it is what the source run predicts.
        distractor_token: The ``r'`` of the metrics, a string that tokenizes to a single token or a token
            id. Unset, it is what the target run predicts.

    Raises:
        ValueError: If the config has no patch element, names a patch type, metric or comparison nothing
            implements, or reads or scores against the source run without a source prompt.
    """

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
    _batch_width: int | None = field(init=False, default=None)

    _index: list[Index[PatchingElementConfig]] | None = field(init=False, default=None)

    _baselines: dict[tuple[str, PatchingMetric], torch.Tensor] = field(init=False, default_factory=dict)
    _source_probabilities: torch.Tensor | None = field(init=False, default=None)
    _noise_scales: dict[str, torch.Tensor] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self._check_available()

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
        try:
            broadcast_length(*map(len, index))
        except ValueError:
            lengths = " and ".join(str(length) for length in sorted({len(idx) for idx in index} - {1}))
            raise ValueError(
                f"Patch elements of {lengths} rows cannot be applied together: patched run i applies row i of every "
                "element, so they need as many rows, or one"
            ) from None
        self._index = index

        return self

    def prepare(self, model: Model, batch_size: int = 1) -> "PatchingInput":
        """Run both prompts, caching what the index reads, as wide as the patched runs will be.

        A backend may round differently at different batch shapes (MPS does, by up to half a logit in
        bfloat16), so a patched run has to be compared with, and patched from, runs of its own shape:
        only then does a patch that writes back what was there reproduce the target run exactly. Each
        prompt is therefore run ``batch_width`` copies abreast and kept at its first row.

        Args:
            model: The model to run.
            batch_size: How many patched runs each forward pass holds.

        Returns:
            The input itself, prepared in place.

        Raises:
            ValueError: If the answer tokens cannot be resolved, or resolve to the same token while a
                difference metric is asked for.
        """
        self.prepare_index(model)

        batch_width = max(1, min(batch_size, len(self.rows())))
        if self._target_cache is None or batch_width != self._batch_width:
            names = self.activation_names()
            with torch.no_grad():
                if self._source_tokens is not None:
                    self._source_logits, self._source_cache = self._run_with_cache(
                        model, self._source_tokens, batch_width, names
                    )
                self._target_logits, self._target_cache = self._run_with_cache(
                    model, self._target_tokens, batch_width, names
                )
            self._batch_width = batch_width

            self._baselines = {}
            self._source_probabilities = None
            self._noise_scales = {}

            if self._source_logits is not None:
                self._source_generation = int(self._source_logits[0, -1].argmax())
            self._target_generation = int(self._target_logits[0, -1].argmax())
            self._resolve_answer_tokens(model)

        return self

    @staticmethod
    def _run_with_cache(
        model: Model, tokens: Int[torch.Tensor, "batch pos"], batch_width: int, names: list[str]
    ) -> tuple[Float[torch.Tensor, "batch pos d_vocab"], ActivationCache]:
        logits, cache = model.run_with_cache(torch.cat([tokens] * batch_width), names_filter=names)

        if batch_width != 1:
            logits = logits[:1].clone()
            cache = ActivationCache(
                {name: activation[:1].clone() for name, activation in cache.cache_dict.items()}, cache.model
            )
        return logits, cache

    def rows(self) -> list[tuple]:
        """Group the index's rows as a run patches them.

        Returns:
            One tuple per patched run, in result row order, holding that run's row of every element.
        """
        return zip_broadcast(*self.index)

    def activation_names(self) -> list[str]:
        """Name every activation the run touches, on either side of the patch.

        The target's own names are read only by the gradient methods, which need the clean activation a
        patch would overwrite and the value they interpolate towards. The two sides name the same
        activation in the common case, so caching both usually costs nothing.

        Returns:
            The hook names, sorted.
        """
        names: set[str] = set()
        for idx in self.index:
            names |= idx.source_activation_names() | idx.target_activation_names()
        return sorted(names)

    def target_activation_names(self) -> list[str]:
        """Name every activation a patch writes.

        Returns:
            The hook names, sorted.
        """
        return sorted(set().union(*(idx.target_activation_names() for idx in self.index))) if self.index else []

    def _check_available(self) -> None:
        """Resolve every element's strategies, metric and comparison, and reject those needing a missing source run.

        Resolved as the input is built, so a config naming something no strategy or metric implements
        fails before a model is involved. Each is collected into a dict, so a type or comparison named
        several times is reported once.

        Raises:
            ValueError: If something named is not implemented, or needs the source run without a source
                prompt.
        """
        if not self.config.patch:
            raise ValueError("A patching run needs at least one patch element")
        sources = {element.source.type: SourceStrategy.get(element.source.type) for element in self.config.patch}
        for element in self.config.patch:
            TargetStrategy.get(element.target.type)
        metrics = {metric: AbstractMetric.get(metric) for metric in self.config.metric}
        comparisons = {
            comparison: AbstractComparison.get(comparison)
            for comparisons in self.config.metric.values()
            for comparison in comparisons
        }
        if self.source_prompt is not None:
            return

        reading = [named.value for named, strategy in sources.items() if strategy.needs_source_run]
        if reading:
            raise ValueError(f"{', '.join(reading)} reads the source run, which needs a source prompt")
        scoring = [
            named.value
            for named, implementation in (*metrics.items(), *comparisons.items())
            if implementation.needs_source_run
        ]
        if scoring:
            raise ValueError(
                f"{', '.join(scoring)} scores the patched run against the source run, which needs a source prompt"
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
                "Without a source prompt there is no run to read the correct token off - pass 'correct_token'"
            )
        return fallback

    @property
    def source_tokens(self) -> Int[torch.Tensor, "batch pos"]:
        """The source prompt's tokens, once the index is prepared."""
        assert self._source_tokens is not None
        return self._source_tokens

    @property
    def target_tokens(self) -> Int[torch.Tensor, "batch pos"]:
        """The target prompt's tokens, once the index is prepared."""
        assert self._target_tokens is not None
        return self._target_tokens

    @property
    def source_logits(self) -> Float[torch.Tensor, "batch pos d_vocab"]:
        """The unpatched source run's logits, once the input is prepared."""
        assert self._source_logits is not None
        return self._source_logits

    @property
    def target_logits(self) -> Float[torch.Tensor, "batch pos d_vocab"]:
        """The unpatched target run's logits, once the input is prepared."""
        assert self._target_logits is not None
        return self._target_logits

    @property
    def source_generation(self) -> int:
        """The token the unpatched source run predicts, once the input is prepared."""
        assert self._source_generation is not None
        return self._source_generation

    @property
    def target_generation(self) -> int:
        """The token the unpatched target run predicts, once the input is prepared."""
        assert self._target_generation is not None
        return self._target_generation

    @property
    def correct_token_id(self) -> int:
        """The id of the correct token ``r``, once the input is prepared."""
        assert self._correct_token_id is not None
        return self._correct_token_id

    @property
    def distractor_token_id(self) -> int:
        """The id of the distractor token ``r'``, once the input is prepared."""
        assert self._distractor_token_id is not None
        return self._distractor_token_id

    @property
    def source_cache(self) -> ActivationCache:
        """The source run's activations, once the input is prepared."""
        assert self._source_cache is not None
        return self._source_cache

    @property
    def target_cache(self) -> ActivationCache:
        """The target run's activations, once the input is prepared."""
        assert self._target_cache is not None
        return self._target_cache

    @property
    def index(self) -> list[Index[PatchingElementConfig]]:
        """One index per patch element, once the index is prepared."""
        assert self._index is not None
        return self._index

    @property
    def layout_index(self) -> Index[PatchingElementConfig]:
        """The element index the result is laid out along: the first with the most rows, so every run has a row."""
        return max(self.index, key=len)

    @property
    def batch_width(self) -> int:
        """How many rows every patched run holds, the width the cached runs were made at."""
        assert self._batch_width is not None
        return self._batch_width

    def source_baseline(self, metric: PatchingMetric) -> torch.Tensor:
        """Read a metric off the unpatched source run.

        Args:
            metric: The metric.

        Returns:
            The metric's value, computed once.
        """
        return self._baseline("source", metric)

    def target_baseline(self, metric: PatchingMetric) -> torch.Tensor:
        """Read a metric off the unpatched target run.

        Args:
            metric: The metric.

        Returns:
            The metric's value, computed once.
        """
        return self._baseline("target", metric)

    def _baseline(self, run: str, metric: PatchingMetric) -> torch.Tensor:
        """Read a metric off one of the two unpatched runs, which a patched run is reported against.

        Args:
            run: ``"source"`` or ``"target"``.
            metric: The metric.

        Returns:
            The metric's value, computed once.
        """
        if (run, metric) not in self._baselines:
            logits = self.source_logits if run == "source" else self.target_logits
            self._baselines[(run, metric)] = AbstractMetric.get(metric).compute(self, logits[0, -1])
        return self._baselines[(run, metric)]

    @property
    def source_probabilities(self) -> torch.Tensor:
        """The unpatched source run's next-token distribution, which ``kl_divergence`` scores against."""
        if self._source_probabilities is None:
            self._source_probabilities = self.source_logits[0, -1].softmax(dim=-1)
        return self._source_probabilities

    def noise_scale(self, name: str) -> torch.Tensor:
        """Find the standard deviation ``gaussian_noise`` draws at for an activation.

        Measured once per activation rather than once per row that noises it.

        Args:
            name: The activation's hook name.

        Returns:
            Three times the target run's own spread there, as in causal tracing (Meng et al. 2022).
        """
        if name not in self._noise_scales:
            self._noise_scales[name] = 3 * self.target_cache[name].std()
        return self._noise_scales[name]

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
        self._baselines = {key: baseline.to(device) for key, baseline in self._baselines.items()}
        self._noise_scales = {name: scale.to(device) for name, scale in self._noise_scales.items()}

        return self
