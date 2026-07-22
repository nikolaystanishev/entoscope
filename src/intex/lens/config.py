from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from intex.model.config import ModelLocation, ModelLocatorConfig
from intex.operation.config import AbstractConfig


class LensType(str, Enum):
    LOGIT_LENS = "logit_lens"
    TUNED_LENS = "tuned_lens"
    PARAMETER_LENS = "parameter_lens"


class ProjectionSpace(str, Enum):
    EMBEDDING = "embedding"
    UNEMBEDDING = "unembedding"


@dataclass(kw_only=True)
class LensConfig(AbstractConfig):
    type: LensType
    top_k: int = 1
    source: ModelLocatorConfig
    translators: str | None = None
    projection: ProjectionSpace | None = None

    def __post_init__(self):
        if self.translators is not None and self.type is not LensType.TUNED_LENS:
            raise ValueError(f"'translators' is a tuned lens setting; it cannot be used with '{self.type.value}'")
        if self.projection is not None and self.type is not LensType.PARAMETER_LENS:
            raise ValueError(f"'projection' is a parameter lens setting; it cannot be used with '{self.type.value}'")
        if self.type is LensType.PARAMETER_LENS and (
            ModelLocation.POSITION in self.source.location
            or any(ModelLocation.POSITION in coordinate for coordinate in self.source.coordinates or [])
        ):
            raise ValueError(
                "'position' is not a parameter lens location: it reads the model's weights, which have no position axis"
            )
