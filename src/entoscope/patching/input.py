import logging
import numbers
from dataclasses import dataclass, field

import torch
from jaxtyping import Float, Int
from transformer_lens import ActivationCache

from entoscope.model import Model
from entoscope.model.activation import check_transformer_activations
from entoscope.model.index import Index
from entoscope.operation.input import OperationInput
from entoscope.patching.config import PatchingConfig, PatchingElementConfig
from entoscope.patching.metric import AbstractComparison, AbstractMetric
from entoscope.patching.processor import SourceStrategy, TargetStrategy
from entoscope.util.collections import broadcast_length, zip_broadcast

logger = logging.getLogger(__name__)


@dataclass
class PatchingInput(OperationInput):
    """The two prompts a patching run sits between, and the tokens its metrics read.

    The target run is the one patched, and the source run is what ``replace`` and ``value_scale`` read.
    A source element that reads no source run, a value or noise on the target's own activation, counts
    its positions in the target prompt.

    Attributes:
        config: The patching config.
        source_prompt: The prompt patched from. ``None`` when no element reads it, and then
            ``kl_divergence`` and ``normalized``, which score against the source run, are unavailable.
        target_prompt: The prompt patched into.
        tokens: The tokens the config's metrics read, by the names the metrics give them, each a string
            that tokenizes to a single token or a token id. A string of several tokens is read as its
            first, with a warning. Only the tokens a metric reads are checked and resolved, so one mapping
            can serve configs that read different ones.

    Raises:
        ValueError: If the config has no patch element, names a patch type, metric or comparison nothing
            implements, reads or scores against the source run without a source prompt, or a metric
            names other than as many tokens as its type reads, or a token that ``tokens`` does not name
            or that is neither a string nor a token id.
    """

    config: PatchingConfig
    source_prompt: str | None
    target_prompt: str
    tokens: dict[str, str | int] = field(default_factory=dict)

    _source_tokens: Int[torch.Tensor, "batch pos"] | None = field(init=False, default=None)
    _target_tokens: Int[torch.Tensor, "batch pos"] | None = field(init=False, default=None)
    _source_logits: Float[torch.Tensor, "batch pos d_vocab"] | None = field(init=False, default=None)
    _target_logits: Float[torch.Tensor, "batch pos d_vocab"] | None = field(init=False, default=None)
    _source_generation: int | None = field(init=False, default=None)
    _target_generation: int | None = field(init=False, default=None)
    _token_ids: dict[str, int] | None = field(init=False, default=None)
    _source_cache: ActivationCache | None = field(init=False, default=None)
    _target_cache: ActivationCache | None = field(init=False, default=None)
    _batch_width: int | None = field(init=False, default=None)

    _index: list[Index[PatchingElementConfig]] | None = field(init=False, default=None)

    _metrics: dict[str, AbstractMetric] | None = field(init=False, default=None)
    _noise_scales: dict[str, torch.Tensor] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self._check_available()
        self._check_tokens()
        if not self.config.metric:
            logger.warning("The config names no metric, so the run reports none and its page draws no plot")

    def prepare(self, model: Model, batch_size: int = 1, *, index_only: bool = False) -> "PatchingInput":
        """Resolve the index and the tokens, then run both prompts, caching what it reads, as wide as the patched runs.

        A backend may round differently at different batch shapes (MPS does, by up to half a logit in
        bfloat16), so a patched run has to be compared with, and patched from, runs of its own shape:
        only then does a patch that writes back what was there reproduce the target run exactly. Each
        prompt is therefore run ``batch_width`` copies abreast and kept at its first row.

        Args:
            model: The model to run.
            batch_size: How many patched runs each forward pass holds.
            index_only: Whether to stop after the index and the tokens, before any forward pass.

        Returns:
            The input itself, prepared in place.

        Raises:
            ValueError: If the elements' row counts cannot be paired, the model computes no transformer activation
                an element names, or a token a metric reads is a string that holds no token or an id outside the
                vocabulary.
        """
        if self._index is None:
            if self.source_prompt is not None:
                self._source_tokens = model.to_tokens(self.source_prompt)
            self._target_tokens = model.to_tokens(self.target_prompt)

            index = [
                Index[PatchingElementConfig](
                    patching_config.source,
                    patching_config.source.parse(model, self._read_tokens(patching_config.source).shape[1]),
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
                    f"Patch elements of {lengths} rows cannot be applied together: patched run i applies row i of "
                    "every element, so they need as many rows, or one"
                ) from None
            check_transformer_activations(model, {name for idx in index for name in idx.activation_names()})
            self._index = index
        if self._token_ids is None:
            self._token_ids = self._resolve_tokens(model)

        if index_only:
            return self

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
            self._noise_scales = {}
            self._metrics = None

            if self._source_logits is not None:
                self._source_generation = int(self._source_logits[0, -1].argmax())
            self._target_generation = int(self._target_logits[0, -1].argmax())

        return self

    def _read_tokens(self, source: PatchingElementConfig) -> Int[torch.Tensor, "batch pos"]:
        """Pick the prompt a source element's positions count in: the run its strategy reads.

        Args:
            source: A source element.

        Returns:
            The source prompt's tokens for an element that reads the source run, and otherwise the
            target prompt's, whose run a noise source corrupts and a value source is written into.
        """
        return self.source_tokens if SourceStrategy.get(source.type).needs_source_run else self.target_tokens

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
            The transformer activation names, sorted.
        """
        return sorted({name for idx in self.index for name in idx.activation_names()})

    def target_activation_names(self) -> list[str]:
        """Name every activation a patch writes.

        Returns:
            The transformer activation names, sorted.
        """
        return sorted({name for idx in self.index for name in idx.target_activation_names()})

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
        metrics = {metric.type: AbstractMetric.get(metric.type) for metric in self.config.metric.values()}
        comparisons = {
            metric.comparison: AbstractComparison.get(metric.comparison) for metric in self.config.metric.values()
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

    def _check_tokens(self) -> None:
        """Reject a metric that names other than as many tokens as its type reads, or one the input does not name.

        A token is a string or an id, so a bool or a float, which would truncate to an unrelated id, is
        refused.

        Raises:
            ValueError: If a metric names too many or too few tokens, one missing from ``tokens``, or one
                that is neither a string nor a token id.
        """
        for name, metric in self.config.metric.items():
            n_tokens = AbstractMetric.get(metric.type).n_tokens
            if len(metric.tokens) != n_tokens:
                raise ValueError(
                    f"{name} is a {metric.type.value}, which reads {n_tokens} token{'' if n_tokens == 1 else 's'}, "
                    f"got {metric.tokens}"
                )
            for token in metric.tokens:
                if token not in self.tokens:
                    named = ", ".join(map(repr, self.tokens)) or "none"
                    raise ValueError(
                        f"{name} reads the token {token!r}, which the input's tokens do not name: they name {named}"
                    )
                value = self.tokens[token]
                if isinstance(value, bool) or not isinstance(value, str | numbers.Integral):
                    raise ValueError(f"The token {token!r} is {value!r}, which is neither a string nor a token id")

    def _resolve_tokens(self, model: Model) -> dict[str, int]:
        """Find the id of every token a metric reads, which needs the tokenizer and no forward pass.

        Args:
            model: The model the tokens are read off.

        Returns:
            Each token's id, by its name.

        Raises:
            ValueError: If a string holds no token, or an id is outside the vocabulary.
        """
        names = dict.fromkeys(token for metric in self.config.metric.values() for token in metric.tokens)
        return {name: self._token_id(model, name) for name in names}

    def _token_id(self, model: Model, name: str) -> int:
        token = self.tokens[name]
        if isinstance(token, str):
            # Tokenized here rather than through ``to_single_token``, whose backends fail with a bare
            # assertion on a string of several tokens.
            ids = model.to_tokens(token, prepend_bos=False)[0]
            if len(ids) == 0:
                raise ValueError(f"The token {name!r} is {token!r}, which holds no token")
            if len(ids) > 1:
                logger.warning(
                    "The token %r is %r, which is %d tokens: its first, %r, is read",
                    name,
                    token,
                    len(ids),
                    model.to_single_str_token(int(ids[0])),
                )
            return int(ids[0])
        if not 0 <= token < model.cfg.d_vocab_out:
            raise ValueError(
                f"The token {name!r} is {token}, outside the model's vocabulary of {model.cfg.d_vocab_out} tokens"
            )
        return int(token)

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

    @property
    def metrics(self) -> dict[str, AbstractMetric]:
        """Every metric of the config by its name, built on the ids of the tokens it reads when first read.

        Built again once the prompts are re-run, or the input moved, so what a metric keeps of the unpatched
        runs is always of the runs the input holds.
        """
        if self._metrics is None:
            assert self._token_ids is not None
            self._metrics = {
                name: AbstractMetric.get(metric.type)([self._token_ids[token] for token in metric.tokens])
                for name, metric in self.config.metric.items()
            }
        assert self._metrics is not None
        return self._metrics

    def noise_scale(self, name: str) -> torch.Tensor:
        """Find the standard deviation ``gaussian_noise`` draws at for an activation.

        Measured once per activation rather than once per row that noises it.

        Args:
            name: The transformer activation name.

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
        self._noise_scales = {name: scale.to(device) for name, scale in self._noise_scales.items()}
        self._metrics = None

        return self
