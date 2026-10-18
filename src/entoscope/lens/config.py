from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from entoscope.model.config import ModelAxis, ModelLocatorConfig
from entoscope.operation.config import AbstractConfig
from entoscope.operation.method import MethodConfig

if TYPE_CHECKING:
    from entoscope.lens.method.factory import LensMethodFactory


@dataclass(kw_only=True)
class LensMethodConfig(MethodConfig):
    """The method config of a lens that takes no fields, such as ``logit_lens``.

    A lens with fields of its own names a dataclass extending this one.
    """

    @classmethod
    def factory(cls) -> type[LensMethodFactory]:
        """Name the factory lens methods register with.

        Returns:
            ``LensMethodFactory``.
        """
        from entoscope.lens.method.factory import LensMethodFactory

        return LensMethodFactory


@dataclass(kw_only=True)
class LensConfig(AbstractConfig):
    """What a lens run reads and how.

    In a YAML file ``method`` is the mapping read strictly into the lens's method config class, or just
    the lens's type for its defaults, so a file naming a lens nothing registered or a field the lens
    does not have fails as it is read.

    Attributes:
        method: The lens's method config, whose ``type`` names the lens.
        top_k: How many tokens to keep per row.
        source: Where to read. A lens that reads activations reads one position per row, and one that
            reads weights reads no position.

    Raises:
        ValueError: If ``top_k`` is below 1, or ``source`` does not name the positions the lens reads.
    """

    method: LensMethodConfig
    top_k: int = 1
    source: ModelLocatorConfig

    def __post_init__(self):
        super().__post_init__()
        if self.top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {self.top_k}")

        source = self.source
        if not LensMethodConfig.factory().get(self.method.type).reads_activations:
            if ModelAxis.POSITION in source.location or any(
                ModelAxis.POSITION in coordinate for coordinate in source.coordinates or []
            ):
                raise ValueError(
                    f"'position' is not a location of '{self.method.type}': it reads the model's weights, "
                    "which have no position axis"
                )
            return

        if source.coordinates is not None:
            one_position_per_row = all(
                isinstance(coordinate.get(ModelAxis.POSITION), int) for coordinate in source.coordinates
            )
        else:
            one_position_per_row = ModelAxis.POSITION in source.location and ModelAxis.POSITION not in (
                *source.group,
                *source.incremental,
                *source.combination.location,
            )
        if not one_position_per_row:
            raise ValueError(
                f"'{self.method.type}' ranks the tokens of one position per row, so its locator names a single "
                "'position' in every coordinate, or has 'position' in 'location' without grouping it"
            )
