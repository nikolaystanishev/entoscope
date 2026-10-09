from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from intex.operation.config import AbstractConfig
from intex.util.data import from_dict, to_dict


@dataclass(kw_only=True)
class CircuitSettings:
    """The settings of a method that takes none; a method with settings names a dataclass extending this one."""


@dataclass(kw_only=True)
class CircuitConfig(AbstractConfig):
    """Configuration for the Circuit Discovery operation.

    Unlike the per-location operations (Lens, Probing, Patching), circuit discovery is
    global: a method builds a graph spanning every layer, token and head, so there is no
    ``ModelLocatorConfig``/``Index`` here - only ``method``, naming the method, and ``settings``, its
    own, given as an instance of its ``settings_class`` or as the mapping a YAML file gives, and read
    strictly into that class as the config is built.

    Whether an interactive visualization is written is chosen by the collector, not here:
    ``CircuitVisualizationCollector`` renders HTML, the base ``CircuitCollector`` does not.
    """

    method: str
    settings: CircuitSettings | dict[str, Any] = field(default_factory=CircuitSettings)

    def __post_init__(self):
        # Imported here: the factory imports the methods, which import this module.
        from intex.circuit.method.factory import CircuitMethodFactory

        settings_class = CircuitMethodFactory.get(self.method).settings_class
        # Read through its raw shape, so the empty default takes the method's defaults and another method's
        # settings - which also extend CircuitSettings - cannot pass as this method's.
        self.settings = from_dict(settings_class, to_dict(self.settings))
