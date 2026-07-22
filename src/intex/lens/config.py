from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from intex.model.config import ModelLocatorConfig
from intex.operation.config import AbstractConfig


class LensType(str, Enum):
    LOGIT_LENS = "logit_lens"
    TUNED_LENS = "tuned_lens"


@dataclass(kw_only=True)
class LensConfig(AbstractConfig):
    type: LensType
    top_k: int = 1
    source: ModelLocatorConfig
    translators: str | None = None

    def __post_init__(self):
        if self.translators is not None and self.type is not LensType.TUNED_LENS:
            raise ValueError(f"'translators' is a tuned lens setting; it cannot be used with '{self.type.value}'")
