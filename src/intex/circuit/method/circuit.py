import abc
from dataclasses import dataclass

import networkx as nx
import torch
from jaxtyping import Int
from transformer_lens import HookedTransformer

from intex.circuit.config import CircuitConfig


@dataclass
class PromptCircuit:
    """One prompt's discovered circuit.

    graph: full thresholded contribution graph, with a separate attention edge per head.
    routes: pruned subgraph of the edges on paths leading to the traced prediction position(s).
    """

    prompt: str
    str_tokens: list[str]
    graph: nx.MultiDiGraph
    routes: nx.MultiDiGraph


class AbstractCircuitMethod(abc.ABC):
    def __init__(self, model: HookedTransformer, config: CircuitConfig):
        self.model = model
        self.config = config

    @abc.abstractmethod
    def build(self, prompt: str, tokens: Int[torch.Tensor, "batch pos"]) -> PromptCircuit:
        """Discover the circuit for one tokenized prompt and return its graphs."""
        raise NotImplementedError
