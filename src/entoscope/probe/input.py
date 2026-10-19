from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Int

from entoscope.model import Model
from entoscope.model.config import POSITION_AXES, ModelLocatorConfig
from entoscope.model.index import Index
from entoscope.operation.input import OperationInput
from entoscope.probe.config import ProbeConfig
from entoscope.probe.pool import pool


@dataclass
class ProbeInput(OperationInput):
    """Positive and negative example prompts whose pooled activations become probe training data.

    The location grid is parsed once. When every prompt tokenizes to the same length the grid uses
    that length, so a filtered position resolves to a concrete index. When prompt lengths differ the
    grid is parsed without one: explicit positions stay symbolic, a non-negative one counting from the
    start and a negative one from each prompt's own end, and a grid that would enumerate every
    position (an unfiltered ``position``, ``dest_pos`` or ``src_pos`` axis) is rejected because its
    size would depend on the prompt. Position axes left outside the grid stay full slices and adapt to
    each prompt.

    ``model.to_tokens`` prepends BOS, so mean or sum pooling over positions includes it while ``last``
    pooling (or a ``-1`` position) reads the final content token.

    Attributes:
        config: The probing config.
        positive_prompts: The examples labeled 1.
        negative_prompts: The examples labeled 0.
    """

    config: ProbeConfig
    positive_prompts: list[str]
    negative_prompts: list[str]

    _labels: Int[torch.Tensor, " examples"] | None = field(init=False, default=None)
    _features: dict[int, Float[torch.Tensor, "examples features"]] | None = field(init=False, default=None)
    _index: Index[ModelLocatorConfig] | None = field(init=False, default=None)
    _token_tensors: list[Int[torch.Tensor, "batch pos"]] | None = field(init=False, default=None)

    def prepare(self, model: Model, *, index_only: bool = False) -> "ProbeInput":
        """Resolve the locations the probes read, then pool every prompt's activation at each into its features.

        Args:
            model: The model the run reads.
            index_only: Whether to stop after the locations, before any forward pass.

        Returns:
            The input itself, prepared in place.

        Raises:
            ValueError: If either set of prompts is empty, an explicit position is out of range for the
                shortest prompt, or a location's feature size varies across prompts.
        """
        if self._index is None:
            if not self.positive_prompts or not self.negative_prompts:
                raise ValueError("Both positive and negative prompts are required to train a probe")

            self._token_tensors = [model.to_tokens(prompt) for prompt in self.positive_prompts + self.negative_prompts]
            lengths = {tokens.shape[1] for tokens in self._token_tensors}

            n_positions = next(iter(lengths)) if len(lengths) == 1 else None
            source = self.config.source.parse(model, n_positions)
            self._validate_positions(source.df, min(lengths))
            self._index = Index[ModelLocatorConfig](self.config.source, source)

        if not index_only and self._features is None:
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
    ) -> dict[int, Float[torch.Tensor, "examples features"]]:
        vectors: dict[int, list[torch.Tensor]] = defaultdict(list)
        for tokens in token_tensors:
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=sorted(index.source_activation_names()))

            for single_index in index:
                activation = single_index.source.read_slices(cache[single_index.source.transformer_activation()])
                assert activation.shape[0] == 1
                feature = pool(activation[0], single_index.source.remaining_axes(), self.config.pool)
                vectors[single_index.output].append(feature.detach().to("cpu", torch.float32))

        features: dict[int, Float[torch.Tensor, "examples features"]] = {}
        for output, feature_list in vectors.items():
            if len({feature.shape[0] for feature in feature_list}) != 1:
                raise ValueError(
                    f"Feature size varies across prompts for index row {output}; a position axis left in the "
                    "features, neither pinned nor pooled, requires equal-length prompts"
                )
            features[output] = torch.stack(feature_list)

        return features

    @staticmethod
    def _validate_positions(index_df: pd.DataFrame, min_length: int) -> None:
        """Check that every explicit position lands inside the shortest prompt, counted from either end.

        Args:
            index_df: The resolved locations.
            min_length: The shortest prompt's token count.

        Raises:
            ValueError: If a value of a position axis is out of range.
        """
        for axis in POSITION_AXES:
            if axis not in index_df.columns:
                continue
            for cell in index_df[axis]:
                for position in cell if isinstance(cell, list) else [cell]:
                    if isinstance(position, (int, np.integer)) and not -min_length <= position < min_length:
                        raise ValueError(
                            f"'{axis.value}' {int(position)} is out of range for the shortest prompt "
                            f"({min_length} tokens)"
                        )

    @property
    def labels(self) -> Int[torch.Tensor, " examples"]:
        """Every example's label, positives first, once the input is prepared."""
        assert self._labels is not None
        return self._labels

    @property
    def features(self) -> dict[int, Float[torch.Tensor, "examples features"]]:
        """Each location's features, one row per example, keyed by result row, once the input is prepared."""
        assert self._features is not None
        return self._features

    @property
    def index(self) -> Index[ModelLocatorConfig]:
        """The locations the probes read, once the index is prepared."""
        assert self._index is not None
        return self._index

    @property
    def token_tensors(self) -> list[Int[torch.Tensor, "batch pos"]]:
        """Every example's tokens, positives first, once the index is prepared."""
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
