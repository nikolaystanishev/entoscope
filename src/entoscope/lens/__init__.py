"""Importing the package registers its built-in lenses, so a config can name one whatever was imported first."""

from entoscope.lens.collector import LensCollector, LensVisualizationCollector
from entoscope.lens.config import LensConfig, LensMethodConfig
from entoscope.lens.input import LensInput
from entoscope.lens.lens import Lens
from entoscope.lens.method.factory import LensMethodFactory
from entoscope.lens.method.lens import AbstractLensMethod
from entoscope.lens.method.logit_lens import LogitLens
from entoscope.lens.method.parameter_lens import ParameterLens, ParameterLensConfig, ProjectionSpace
from entoscope.lens.method.tuned_lens import TunedLens, TunedLensConfig

__all__ = [
    "AbstractLensMethod",
    "Lens",
    "LensCollector",
    "LensConfig",
    "LensInput",
    "LensMethodConfig",
    "LensMethodFactory",
    "LensVisualizationCollector",
    "LogitLens",
    "ParameterLens",
    "ParameterLensConfig",
    "ProjectionSpace",
    "TunedLens",
    "TunedLensConfig",
]
