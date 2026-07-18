from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from intex.model.config import ModelLocatorConfig
from intex.operation.config import AbstractConfig


class LensType(str, Enum):
    LOGIT_LENS = "logit_lens"


@dataclass(kw_only=True)
class LensConfig(AbstractConfig):
    type: LensType
    top_k: int = 1
    source: ModelLocatorConfig
