import logging
from importlib.metadata import version

from entoscope.circuit import (
    CircuitConfig,
    CircuitDiscoverer,
    CircuitInput,
    CircuitMethodConfig,
    InformationFlowRoutesConfig,
)
from entoscope.lens import (
    Lens,
    LensConfig,
    LensInput,
    LensMethodConfig,
    ParameterLensConfig,
    ProjectionSpace,
    TunedLensConfig,
)
from entoscope.model import (
    CachedModel,
    Combination,
    Coordinates,
    Filter,
    Group,
    Incremental,
    ModelLayerLocation,
    ModelLocation,
    ModelLocatorConfig,
)
from entoscope.operation import ConfigFactory, from_dict, to_dict
from entoscope.patching import (
    IntegratedGradientsConfig,
    MetricComparison,
    Patcher,
    PatchingConfig,
    PatchingElementConfig,
    PatchingInput,
    PatchingMethodConfig,
    PatchingMetric,
    PatchingType,
    SinglePatchingConfig,
)
from entoscope.probe import (
    LinearProbeConfig,
    PoolingType,
    ProbeConfig,
    ProbeInput,
    ProbeMethodConfig,
    ProbeMetric,
    Prober,
)
from entoscope.script import experiment

__version__ = version("entoscope")

__all__ = [
    "CachedModel",
    "CircuitConfig",
    "CircuitDiscoverer",
    "CircuitInput",
    "CircuitMethodConfig",
    "Combination",
    "ConfigFactory",
    "Coordinates",
    "Filter",
    "Group",
    "Incremental",
    "InformationFlowRoutesConfig",
    "IntegratedGradientsConfig",
    "Lens",
    "LensConfig",
    "LensInput",
    "LensMethodConfig",
    "LinearProbeConfig",
    "MetricComparison",
    "ModelLayerLocation",
    "ModelLocation",
    "ModelLocatorConfig",
    "ParameterLensConfig",
    "Patcher",
    "PatchingConfig",
    "PatchingElementConfig",
    "PatchingInput",
    "PatchingMethodConfig",
    "PatchingMetric",
    "PatchingType",
    "PoolingType",
    "ProbeConfig",
    "ProbeInput",
    "ProbeMethodConfig",
    "ProbeMetric",
    "Prober",
    "ProjectionSpace",
    "SinglePatchingConfig",
    "TunedLensConfig",
    "__version__",
    "experiment",
    "from_dict",
    "to_dict",
]

# entoscope owns its output instead of deferring to the root logger: dependencies configure root as a
# side effect (transformer_lens logs through the module-level ``logging.info``, which runs
# ``basicConfig``), and that would reformat entoscope's messages behind the user's back.
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_logger.propagate = False
_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter("[entoscope] %(levelname)s %(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)
_logger.addHandler(_handler)
