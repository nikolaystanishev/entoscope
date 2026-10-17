from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from entoscope.operation.config import AbstractConfig
from entoscope.operation.method import MethodConfig

if TYPE_CHECKING:
    from entoscope.circuit.method.factory import CircuitMethodFactory


@dataclass(kw_only=True)
class CircuitMethodConfig(MethodConfig):
    """The method config of a method that takes no fields; a method with fields names a dataclass extending this
    one."""

    @classmethod
    def factory(cls) -> type[CircuitMethodFactory]:
        from entoscope.circuit.method.factory import CircuitMethodFactory

        return CircuitMethodFactory


@dataclass(kw_only=True)
class CircuitConfig(AbstractConfig):
    """Configuration for the Circuit Discovery operation.

    Unlike the per-location operations (Lens, Probing, Patching), circuit discovery is
    global: a method builds a graph spanning every layer, token and head, so there is no
    ``ModelLocatorConfig``/``Index`` here - only ``method``, the method config, ``type`` naming the
    method: an instance of its method config class, or in a YAML file the mapping read strictly into that class, or
    just the method's type for its defaults.

    Whether an interactive visualization is written is chosen by the collector, not here:
    ``CircuitVisualizationCollector`` renders HTML, the base ``CircuitCollector`` does not.
    """

    method: CircuitMethodConfig
