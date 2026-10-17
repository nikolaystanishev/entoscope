import abc
from dataclasses import dataclass
from typing import Generic, cast

import networkx as nx
import torch
from jaxtyping import Int
from typing_extensions import TypeVar

from entoscope.circuit.config import CircuitConfig, CircuitMethodConfig
from entoscope.model import Model

CircuitMethodConfigT = TypeVar("CircuitMethodConfigT", bound=CircuitMethodConfig, default=CircuitMethodConfig)


@dataclass
class PromptCircuit:
    """One prompt's discovered circuit.

    ``graph`` is the full thresholded contribution graph, with a separate attention edge per head, and
    ``routes`` the subgraph of its edges on paths leading to the traced prediction position(s).

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


class AbstractCircuitMethod(abc.ABC, Generic[CircuitMethodConfigT]):
    """Discovers the circuit behind one prompt at a time; built once per run.

    Its type argument names the dataclass of its method config, extending ``CircuitMethodConfig``, which a config
    file's ``method`` is read into; a method naming none takes ``CircuitMethodConfig`` itself, which no field
    fits.
    """

    def __init__(self, model: Model, config: CircuitConfig):
        self.model = model
        self.config = config
        self.method_config = cast(CircuitMethodConfigT, config.method)

    @abc.abstractmethod
    def build(self, prompt: str, tokens: Int[torch.Tensor, "batch pos"]) -> PromptCircuit:
        """Discover the circuit for one tokenized prompt and return its graphs."""
        raise NotImplementedError

    def activation_names(self) -> list[str] | None:
        """The hooks ``build`` caches, which a plan reports; ``None`` when it caches every activation."""
        return None
