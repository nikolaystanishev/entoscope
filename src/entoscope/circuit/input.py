from dataclasses import dataclass

import torch

from entoscope.circuit.config import CircuitConfig
from entoscope.model import Model
from entoscope.operation.input import OperationInput


@dataclass
class CircuitInput(OperationInput):
    """A list of prompts to discover circuits for; one contribution graph is built per prompt.

    It holds no activations: the method runs each prompt as it builds that prompt's circuit, so peak
    memory stays bounded by the largest single prompt regardless of how many prompts are passed.
    """

    config: CircuitConfig
    prompts: list[str]

    def __post_init__(self):
        if not self.prompts:
            raise ValueError("At least one prompt is required for circuit discovery")

    def prepare_index(self, model: Model) -> "CircuitInput":
        return self

    def to(self, device: str | torch.device = "cpu") -> "CircuitInput":
        return self
