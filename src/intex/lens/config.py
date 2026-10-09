from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from intex.model.config import ModelLocatorConfig
from intex.operation.config import AbstractConfig
from intex.util.data import from_dict, to_dict


@dataclass(kw_only=True)
class LensSettings:
    """The settings of a lens that takes none; a lens with settings names a dataclass extending this one."""


@dataclass(kw_only=True)
class LensConfig(AbstractConfig):
    """``method`` names the lens and ``settings`` are its own, given as an instance of its ``settings_class``
    or as the mapping a YAML file gives, and read strictly into that class as the config is built, so a lens
    nothing registered or a setting the lens does not have fails right away."""

    method: str
    top_k: int = 1
    source: ModelLocatorConfig
    settings: LensSettings | dict[str, Any] = field(default_factory=LensSettings)

    def __post_init__(self):
        if not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {self.top_k}")
        # Imported here: the factory imports the lenses, which import this module.
        from intex.lens.method.factory import LensMethodFactory

        settings_class = LensMethodFactory.get(self.method).settings_class
        # Read through its raw shape, so the empty default takes the lens's defaults and another lens's
        # settings - which also extend LensSettings - cannot pass as this lens's.
        self.settings = from_dict(settings_class, to_dict(self.settings))
