import abc
from dataclasses import dataclass
from typing import ClassVar

import networkx as nx
import torch
from jaxtyping import Int

from intex.circuit.config import CircuitConfig, CircuitSettings
from intex.model import Model


@dataclass
class PromptCircuit:
    """One prompt's discovered circuit.

    graph: full thresholded contribution graph, with a separate attention edge per head.
    routes: pruned subgraph of the edges on paths leading to the traced prediction position(s).

    The circuit page and the GraphML files read the attributes Information Flow Routes writes: each
    node a ``kind`` (``input``, ``after_attn``, ``mlp`` or ``after_ffn``), ``layer``, ``position`` and
    ``token``, and each edge a ``kind`` (``attention``, ``residual`` or ``mlp``), ``weight`` and
    ``head`` (-1 off attention). A method writing other attributes runs with ``CircuitCollector``,
    which draws no page.
    """

    prompt: str
    str_tokens: list[str]
    graph: nx.MultiDiGraph
    routes: nx.MultiDiGraph


class AbstractCircuitMethod(abc.ABC):
    """Discovers the circuit behind one prompt at a time; built once per run.

    ``settings_class`` names the dataclass of its settings, extending ``CircuitSettings``, which the config
    reads its ``settings`` into as it is built; a method naming none takes ``CircuitSettings`` itself, which
    no setting fits.
    """

    settings_class: ClassVar[type[CircuitSettings]] = CircuitSettings

    def __init__(self, model: Model, config: CircuitConfig):
        self.model = model
        self.config = config
        self.settings = config.settings

    @abc.abstractmethod
    def build(self, prompt: str, tokens: Int[torch.Tensor, "batch pos"]) -> PromptCircuit:
        """Discover the circuit for one tokenized prompt and return its graphs."""
        raise NotImplementedError
