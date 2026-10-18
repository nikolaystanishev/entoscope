from dataclasses import dataclass

import torch

from entoscope.circuit.config import CircuitConfig
from entoscope.model import Model
from entoscope.operation.input import OperationInput


@dataclass
class CircuitInput(OperationInput):
    """The prompts to discover circuits for, one contribution graph per prompt.

    It holds no activations: the method runs each prompt as it builds that prompt's circuit, so peak
    memory stays bounded by the largest single prompt regardless of how many prompts are passed.

    Attributes:
        config: The circuit discovery config.
        prompts: The prompts.

    Raises:
        ValueError: If ``prompts`` is empty.
    """

    config: CircuitConfig
    prompts: list[str]

    def __post_init__(self):
        if not self.prompts:
            raise ValueError("At least one prompt is required for circuit discovery")

    def prepare(self, model: Model, *, index_only: bool = False) -> "CircuitInput":
        return self

    def to(self, device: str | torch.device = "cpu") -> "CircuitInput":
        return self
