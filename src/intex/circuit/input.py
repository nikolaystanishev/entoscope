from dataclasses import dataclass, field

import torch
from jaxtyping import Int
from transformer_lens import HookedTransformer

from intex.circuit.config import CircuitConfig
from intex.operation.input import OperationInput


@dataclass
class CircuitInput(OperationInput):
    """A list of prompts to discover circuits for; one contribution graph is built per prompt.

    Only tokenization happens in ``prepare`` — the activation cache is (re)built one prompt at a
    time inside the method, so peak memory stays bounded by the largest single prompt regardless
    of how many prompts are passed (mirrors ``intex.probe.input.ProbeInput``).
    """

    prompts: list[str]
    config: CircuitConfig

    _tokens: list[Int[torch.Tensor, "batch pos"]] | None = field(init=False, default=None)

    def prepare(self, model: HookedTransformer) -> "CircuitInput":
        if self._tokens is None:
            if not self.prompts:
                raise ValueError("At least one prompt is required for circuit discovery")
            self._tokens = [model.to_tokens(prompt) for prompt in self.prompts]

        return self

    @property
    def tokens(self) -> list[Int[torch.Tensor, "batch pos"]]:
        assert self._tokens is not None
        return self._tokens

    def to(self, device: str | torch.device = "cpu") -> "CircuitInput":
        if self._tokens is not None:
            self._tokens = [tokens.to(device) for tokens in self._tokens]

        return self
