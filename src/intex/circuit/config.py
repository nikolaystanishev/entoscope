from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from intex.operation.config import AbstractConfig


class CircuitMethodType(str, Enum):
    INFORMATION_FLOW_ROUTES = "information_flow_routes"


@dataclass(kw_only=True)
class CircuitConfig(AbstractConfig):
    """Configuration for the Circuit Discovery operation.

    Unlike the per-location operations (Lens, Probing, Patching), circuit discovery is
    global: a method builds a graph spanning every layer, token and head, so there is no
    ``ModelLocatorConfig``/``Index`` here.

    threshold: Renormalizing edge threshold applied to contributions while building the
        full graph. Contributions below it are dropped and the rest renormalized so the
        incoming weights of each representation still sum to 1 (see the Information Flow
        Routes paper). ``0.0`` keeps every edge.
    path_threshold: Threshold used when pruning the routes-to-prediction subgraph. Defaults
        to ``threshold`` when ``None``.
    prediction_positions: Token positions whose prediction the routes are traced back from.
        Negative indices count from the end of each prompt. Defaults to ``[-1]`` (the
        next-token prediction).

    Whether an interactive visualization is written is chosen by the collector, not here:
    ``CircuitVisualizationCollector`` renders HTML, the base ``CircuitCollector`` does not.
    """

    type: CircuitMethodType
    threshold: float = 0.04
    path_threshold: float | None = None
    prediction_positions: list[int] = field(default_factory=lambda: [-1])

    def resolved_path_threshold(self) -> float:
        return self.threshold if self.path_threshold is None else self.path_threshold
