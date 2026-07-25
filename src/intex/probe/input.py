from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Int

from intex.model import Model
from intex.model.config import ModelLocation, ModelLocatorConfig
from intex.model.index import Index
from intex.operation.input import OperationInput
from intex.probe.config import ProbeConfig
from intex.probe.pool import pool


@dataclass
class ProbeInput(OperationInput):
    """Positive/negative example prompts whose pooled activations become probe training data.

    The location grid is parsed once. When every prompt tokenizes to the same length the grid uses
    that length, so a filtered position resolves to a concrete index. When prompt lengths differ the
    grid is parsed length-agnostically (``n_positions=None``): explicit positions stay symbolic —
    non-negative counts from the start, negative from each prompt's own end (torch resolves it
    against that prompt's cache at slice time) — and a grid that would enumerate every position (an
    unfiltered ``position`` axis) is rejected because its size would depend on the prompt. Every
    explicit position must be valid for the shortest prompt. Position axes left outside the grid
    stay full slices and adapt to each prompt.

    ``model.to_tokens`` prepends BOS, so mean/sum pooling over positions includes it while ``last``
    pooling (or a ``-1`` position) reads the final content token.
    """

    config: ProbeConfig
    positive_prompts: list[str]
    negative_prompts: list[str]

    _labels: Int[torch.Tensor, " examples"] | None = field(init=False, default=None)
    _features: dict[tuple[int, ...], Float[torch.Tensor, "examples features"]] | None = field(init=False, default=None)
    _index: Index[ModelLocatorConfig] | None = field(init=False, default=None)
    _token_tensors: list[Int[torch.Tensor, "batch pos"]] | None = field(init=False, default=None)

    def prepare_index(self, model: Model) -> "ProbeInput":
        if self._index is not None:
            return self

        if not self.positive_prompts or not self.negative_prompts:
            raise ValueError("Both positive and negative prompts are required to train a probe")

        self._token_tensors = [model.to_tokens(prompt) for prompt in self.positive_prompts + self.negative_prompts]
        lengths = {tokens.shape[1] for tokens in self._token_tensors}

        n_positions = next(iter(lengths)) if len(lengths) == 1 else None
        source = self.config.source.parse(model, n_positions)
        self._validate_positions(source.df, min(lengths))
        self._index = Index[ModelLocatorConfig](self.config.source, source)

        return self

    def prepare(self, model: Model) -> "ProbeInput":
        self.prepare_index(model)

        # Guarded on the features, not the index: a ``plan`` may have resolved the index already.
        if self._features is None:
            self._features = self._collect_features(model, self.token_tensors, self.index)
            self._labels = torch.cat(
                [
                    torch.ones(len(self.positive_prompts), dtype=torch.long),
                    torch.zeros(len(self.negative_prompts), dtype=torch.long),
                ]
            )

        return self

    def _collect_features(
        self,
        model: Model,
        token_tensors: list[Int[torch.Tensor, "batch pos"]],
        index: Index[ModelLocatorConfig],
    ) -> dict[tuple[int, ...], Float[torch.Tensor, "examples features"]]:
        # One prompt at a time keeps a single activation cache alive; features go to CPU
        # immediately, so memory stays bounded regardless of the number of examples.
        vectors: dict[tuple[int, ...], list[torch.Tensor]] = defaultdict(list)
        for tokens in token_tensors:
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=sorted(index.source_activation_names()))

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

        return features

    @staticmethod
    def _validate_positions(index_df: pd.DataFrame, min_length: int) -> None:
        """Every explicit position must land inside the shortest prompt (for both anchors)."""
        if ModelLocation.POSITION not in index_df.columns:
            return
        for cell in index_df[ModelLocation.POSITION]:
            for position in cell if isinstance(cell, list) else [cell]:
                if isinstance(position, (int, np.integer)) and not -min_length <= position < min_length:
                    raise ValueError(
                        f"Position {int(position)} is out of range for the shortest prompt ({min_length} tokens)"
                    )

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

    @property
    def token_tensors(self) -> list[Int[torch.Tensor, "batch pos"]]:
        assert self._token_tensors is not None
        return self._token_tensors

    def to(self, device: str | torch.device = "cpu") -> "ProbeInput":
        if self._labels is not None:
            self._labels = self._labels.to(device)
        if self._features is not None:
            self._features = {output: features.to(device) for output, features in self._features.items()}
        if self._token_tensors is not None:
            self._token_tensors = [tokens.to(device) for tokens in self._token_tensors]

        return self
