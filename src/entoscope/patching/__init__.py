"""Importing the package registers its built-in patching methods, so a config can name one whatever was
imported first."""

from entoscope.patching.collector import PatchingCollector, PatchingVisualizationCollector
from entoscope.patching.config import (
    MetricComparison,
    PatchingConfig,
    PatchingElementConfig,
    PatchingMethodConfig,
    PatchingMetric,
    PatchingType,
    SinglePatchingConfig,
)
from entoscope.patching.input import PatchingInput
from entoscope.patching.method.attribution import (
    AttributionPatchingMethod,
    IntegratedGradientsConfig,
    IntegratedGradientsMethod,
)
from entoscope.patching.method.exact import ExactPatchingMethod
from entoscope.patching.method.factory import PatchingMethodFactory
from entoscope.patching.method.patching import AbstractPatchingMethod
from entoscope.patching.patcher import Patcher

__all__ = [
    "AbstractPatchingMethod",
    "AttributionPatchingMethod",
    "ExactPatchingMethod",
    "IntegratedGradientsConfig",
    "IntegratedGradientsMethod",
    "MetricComparison",
    "Patcher",
    "PatchingCollector",
    "PatchingConfig",
    "PatchingElementConfig",
    "PatchingInput",
    "PatchingMethodConfig",
    "PatchingMethodFactory",
    "PatchingMetric",
    "PatchingType",
    "PatchingVisualizationCollector",
    "SinglePatchingConfig",
]
