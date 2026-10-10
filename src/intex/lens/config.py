from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from intex.model.config import ModelLocation, ModelLocatorConfig
from intex.operation.config import AbstractConfig
from intex.operation.method import MethodConfig

if TYPE_CHECKING:
    from intex.lens.method.factory import LensMethodFactory


@dataclass(kw_only=True)
class LensMethodConfig(MethodConfig):
    """The method config of a lens that takes no fields, such as ``logit_lens``; a lens with fields names a
    dataclass extending this one."""

    @classmethod
    def factory(cls) -> type[LensMethodFactory]:
        # Imported here: the factory imports the lenses, which import this module.
        from intex.lens.method.factory import LensMethodFactory

        return LensMethodFactory


@dataclass(kw_only=True)
class LensConfig(AbstractConfig):
    """``method`` is the lens's method config, ``type`` naming the lens: an instance of its method config class,
    or in a YAML file the mapping read strictly into that class, or just the lens's type for its defaults - so
    a file naming a lens nothing registered or a field the lens does not have fails as it is read."""

    method: LensMethodConfig
    top_k: int = 1
    source: ModelLocatorConfig

    def __post_init__(self):
        if not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {self.top_k}")
        if not LensMethodConfig.factory().get(self.method.type).reads_activations and (
            ModelLocation.POSITION in self.source.location
            or any(ModelLocation.POSITION in coordinate for coordinate in self.source.coordinates or [])
        ):
            raise ValueError(
                f"'position' is not a location of '{self.method.type}': it reads the model's weights, "
                "which have no position axis"
            )
