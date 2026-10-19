from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from entoscope.operation.config import AbstractConfig
from entoscope.operation.method import MethodConfig

if TYPE_CHECKING:
    from entoscope.circuit.method.factory import CircuitMethodFactory


@dataclass(kw_only=True)
class CircuitMethodConfig(MethodConfig):
    """The method config of a circuit discovery method that takes no fields.

    A method with fields of its own names a dataclass extending this one.
    """

    @classmethod
    def factory(cls) -> type[CircuitMethodFactory]:
        """Name the factory circuit discovery methods register with.

        Returns:
            ``CircuitMethodFactory``.
        """
        from entoscope.circuit.method.factory import CircuitMethodFactory

        return CircuitMethodFactory


@dataclass(kw_only=True)
class CircuitConfig(AbstractConfig):
    """What a circuit discovery run builds.

    Unlike the per-location operations (lens, patching and probing), circuit discovery is global: a
    method builds a graph spanning every layer, token and head, so there is no locator here. Whether a
    page is drawn is chosen by the collector, not here.

    Attributes:
        method: The method config, its ``type`` naming the method. In a YAML file it is the mapping read
            strictly into the method's config class, or just the method's type for its defaults.
    """

    method: CircuitMethodConfig
