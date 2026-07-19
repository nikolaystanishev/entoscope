from collections import defaultdict
from dataclasses import dataclass, field

import torch
from jaxtyping import Float, Int
from transformer_lens import HookedTransformer

from intex.model.config import ModelLocatorConfig
from intex.model.index import Index
from intex.operation.input import OperationInput
from intex.probe.config import ProbeConfig
from intex.probe.pool import pool
from intex.util.collections import dataframe_to_list


@dataclass
class ProbeInput(OperationInput):
    """Positive/negative example prompts whose pooled activations become probe training data.

    The probe grid is parsed once at the shared-prefix length (the shortest prompt), so every
    grid position exists in every prompt; unpooled position axes stay full slices and adapt to
    each prompt's own cache. A per-prompt parse only validates that the location config resolves
    to the same grid points for every prompt.

    ``model.to_tokens`` prepends BOS, so mean/sum pooling over positions includes it and
    last pooling reads the final content token.
    """

    positive_prompts: list[str]
    negative_prompts: list[str]
    config: ProbeConfig

    _labels: Int[torch.Tensor, " examples"] | None = field(init=False, default=None)
    _features: dict[tuple[int, ...], Float[torch.Tensor, "examples features"]] | None = field(init=False, default=None)
    _index: Index[ModelLocatorConfig] | None = field(init=False, default=None)

    def prepare(self, model: HookedTransformer) -> "ProbeInput":
        if self._index is not None:
            return self

        if not self.positive_prompts or not self.negative_prompts:
            raise ValueError("Both positive and negative prompts are required to train a probe")

        tokens_list = [model.to_tokens(prompt) for prompt in self.positive_prompts + self.negative_prompts]
        min_positions = min(tokens.shape[1] for tokens in tokens_list)
        index = Index[ModelLocatorConfig](self.config.source, self.config.source.parse(model, min_positions))
        probe_rows = {str(row) for row in dataframe_to_list(index.source.df)}

        vectors: dict[tuple[int, ...], list[torch.Tensor]] = defaultdict(list)
        # One prompt at a time keeps a single activation cache alive; features go to CPU
        # immediately, so memory stays bounded regardless of the number of examples.
        for tokens in tokens_list:
            prompt_rows = {str(row) for row in dataframe_to_list(self.config.source.parse(model, tokens.shape[1]).df)}
            if not probe_rows <= prompt_rows:
                raise ValueError(
                    "Location grid resolves differently across prompts; positions that depend on "
                    "the prompt length (e.g. negative values) require equal-length prompts"
                )

            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)

            for single_index in index:
                activation = cache[single_index.source.transformer_activation()][single_index.source.slices()]
                assert activation.shape[0] == 1
                feature = pool(activation[0], single_index.source.remaining_axes(), self.config.pool)
                vectors[single_index.output].append(feature.detach().to("cpu", torch.float32))

        features: dict[tuple[int, ...], Float[torch.Tensor, "examples features"]] = {}
        for output, feature_list in vectors.items():
            if len({feature.shape[0] for feature in feature_list}) != 1:
                raise ValueError(
                    f"Feature size varies across prompts for location {output}; activations with a "
                    "position-sized feature axis (e.g. 'attn.pattern') require equal-length prompts"
                )
            features[output] = torch.stack(feature_list)

        self._labels = torch.cat(
            [
                torch.ones(len(self.positive_prompts), dtype=torch.long),
                torch.zeros(len(self.negative_prompts), dtype=torch.long),
            ]
        )
        self._features = features
        self._index = index

        return self

    @property
    def labels(self) -> Int[torch.Tensor, " examples"]:
        assert self._labels is not None
        return self._labels

    @property
    def features(self) -> dict[tuple[int, ...], Float[torch.Tensor, "examples features"]]:
        assert self._features is not None
        return self._features

    @property
    def index(self) -> Index[ModelLocatorConfig]:
        assert self._index is not None
        return self._index

    def to(self, device: str | torch.device = "cpu") -> "ProbeInput":
        if self._labels is not None:
            self._labels = self._labels.to(device)
        if self._features is not None:
            self._features = {output: features.to(device) for output, features in self._features.items()}

        return self
