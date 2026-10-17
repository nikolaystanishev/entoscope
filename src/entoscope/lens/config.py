from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from entoscope.model.config import ModelLocation, ModelLocatorConfig
from entoscope.operation.config import AbstractConfig
from entoscope.operation.method import MethodConfig

if TYPE_CHECKING:
    from entoscope.lens.method.factory import LensMethodFactory


@dataclass(kw_only=True)
class LensMethodConfig(MethodConfig):
    """The method config of a lens that takes no fields, such as ``logit_lens``; a lens with fields names a
    dataclass extending this one."""

    @classmethod
    def factory(cls) -> type[LensMethodFactory]:
        from entoscope.lens.method.factory import LensMethodFactory

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

        source = self.source
        if not LensMethodConfig.factory().get(self.method.type).reads_activations:
            if ModelLocation.POSITION in source.location or any(
                ModelLocation.POSITION in coordinate for coordinate in source.coordinates or []
            ):
                raise ValueError(
                    f"'position' is not a location of '{self.method.type}': it reads the model's weights, "
                    "which have no position axis"
                )
            return

        if source.coordinates is not None:
            one_position_per_row = all(
                isinstance(coordinate.get(ModelLocation.POSITION), int) for coordinate in source.coordinates
            )
        else:
            one_position_per_row = ModelLocation.POSITION in source.location and ModelLocation.POSITION not in (
                *source.group,
                *source.incremental,
                *source.combination.location,
            )
        if not one_position_per_row:
            raise ValueError(
                f"'{self.method.type}' ranks the tokens of one position per row, so its locator names a single "
                "'position' in every coordinate, or has 'position' in 'location' without grouping it"
            )
