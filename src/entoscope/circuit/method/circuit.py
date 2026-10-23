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

    The circuit page and the GraphML files read the attributes Information Flow Routes writes: each
    node a ``kind`` (``input``, ``after_attn``, ``mlp`` or ``after_ffn``), ``layer``, ``position`` and
    ``token``, and each edge a ``kind`` (``attention``, ``residual`` or ``mlp``), ``weight`` and
    ``head`` (-1 off attention). A method writing other attributes runs with ``CircuitCollector``,
    which draws no page.

    Attributes:
        prompt: The prompt.
        str_tokens: The prompt's token strings, one per position.
        graph: The full thresholded contribution graph, with a separate attention edge per head.
        routes: The subgraph of the edges on paths leading to the traced prediction positions.
    """

    prompt: str
    str_tokens: list[str]
    graph: nx.MultiDiGraph
    routes: nx.MultiDiGraph


class AbstractCircuitMethod(abc.ABC, Generic[CircuitMethodConfigT]):
    """The base of every circuit discovery method: discovers the circuit behind one prompt at a time.

    A method is built once per run. Its type argument names the dataclass of its method config, extending
    ``CircuitMethodConfig``, which a config file's ``method`` is read into. A method naming none takes
    ``CircuitMethodConfig`` itself, which no field fits.

    Args:
        model: The model the method reads.
        config: The run's config.

    Attributes:
        method_config: The run's method config, typed by the method's type argument.
    """

    def __init__(self, model: Model, config: CircuitConfig):
        self.model = model
        self.config = config
        self.method_config = cast(CircuitMethodConfigT, config.method)

    @abc.abstractmethod
    def build(self, prompt: str, tokens: Int[torch.Tensor, "batch pos"]) -> PromptCircuit:
        """Discover the circuit behind one prompt.

        Args:
            prompt: The prompt.
            tokens: The prompt's tokens, ``[1, pos]``.

        Returns:
            The prompt's circuit.
        """
        raise NotImplementedError

    def activation_names(self) -> list[str] | None:
        """Name the transformer activations ``build`` caches, which a plan reports.

        Returns:
            The transformer activation names, or ``None`` when ``build`` caches every activation, the default.
        """
        return None
