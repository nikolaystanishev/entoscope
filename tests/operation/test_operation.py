"""What every operation's run shares: what it caches, how often it runs a prompt, and what ``to`` moves."""

import pytest
import torch

from entoscope.circuit.config import CircuitConfig
from entoscope.circuit.discoverer import CircuitDiscoverer
from entoscope.circuit.input import CircuitInput
from entoscope.circuit.method.information_flow_routes import InformationFlowRoutesConfig
from entoscope.lens.config import LensConfig, LensMethodConfig
from entoscope.lens.input import LensInput
from entoscope.lens.lens import Lens
from entoscope.model.config import Filter, ModelAxis, ModelLayerLocation, ModelLocatorConfig
from entoscope.patching.collector import PatchingCollector
from entoscope.patching.config import (
    MetricComparison,
    PatchingConfig,
    PatchingElementConfig,
    PatchingMethodConfig,
    PatchingMetric,
    PatchingMetricConfig,
    PatchingType,
    SinglePatchingConfig,
)
from entoscope.patching.input import PatchingInput
from entoscope.patching.patcher import Patcher
from entoscope.probe.config import PoolingType, ProbeConfig, ProbeMetric
from entoscope.probe.input import ProbeInput
from entoscope.probe.method.linear import LinearProbeConfig
from entoscope.probe.prober import Prober

from ..conftest import record_runs

SOURCE_PROMPT = "The capital of France is"
TARGET_PROMPT = "The capital of Spain is"
POSITIVE = [SOURCE_PROMPT, "I loved it"]
NEGATIVE = [TARGET_PROMPT, "I hated it"]
# One layer of a three-layer model, so what a run caches is a small part of what it could.
ONE_LAYER = ModelLocatorConfig(
    activation=ModelLayerLocation.RESIDUAL_POST,
    location=[ModelAxis.LAYER, ModelAxis.POSITION],
    filter=Filter.from_config([{"layer": 1}]),
)


def _lens(model):
    config = LensConfig(method=LensMethodConfig(type="logit_lens"), source=ONE_LAYER)
    return Lens(model), LensInput(config, SOURCE_PROMPT)


def _probe(model):
    config = ProbeConfig(
        method=LinearProbeConfig(type="linear"),
        pool={ModelAxis.POSITION: PoolingType.MEAN},
        metric=[ProbeMetric.ACCURACY],
        source=ModelLocatorConfig(
            activation=ModelLayerLocation.RESIDUAL_POST,
            location=[ModelAxis.LAYER],
            filter=Filter.from_config([{"layer": 1}]),
        ),
    )
    return Prober(model), ProbeInput(config, POSITIVE, NEGATIVE)


def _patching(model):
    # Gaussian noise and a normalized metric make the run fill the input's lazily built tensors too:
    # the noise scale and both prompts' baselines.
    def element(patching_type: PatchingType) -> PatchingElementConfig:
        return PatchingElementConfig(
            activation=ModelLayerLocation.RESIDUAL_POST,
            location=[ModelAxis.LAYER],
            filter=Filter.from_config([{"layer": 1}]),
            type=patching_type,
        )

    config = PatchingConfig(
        metric={
            "logit_diff": PatchingMetricConfig(
                type=PatchingMetric.LOGIT_DIFF,
                tokens=["correct", "distractor"],
                comparison=MetricComparison.NORMALIZED,
            )
        },
        method=PatchingMethodConfig(type="exact"),
        patch=[SinglePatchingConfig(source=element(PatchingType.GAUSSIAN_NOISE), target=element(PatchingType.REPLACE))],
    )
    patching_input = PatchingInput(
        config, SOURCE_PROMPT, TARGET_PROMPT, tokens={"correct": " Paris", "distractor": " Madrid"}
    )
    return Patcher(model, collector_class=PatchingCollector), patching_input


def _circuit(model):
    config = CircuitConfig(method=InformationFlowRoutesConfig(type="information_flow_routes"))
    return CircuitDiscoverer(model), CircuitInput(config, [SOURCE_PROMPT])


# How many prompts each input runs through the model before its operation reads it.
CACHED_PROMPTS = {_lens: 1, _probe: len(POSITIVE) + len(NEGATIVE), _patching: 2}
IDS = {_lens: "lens", _probe: "probe", _patching: "patching", _circuit: "circuit"}


@pytest.mark.parametrize("operation", list(IDS), ids=list(IDS.values()))
def test_a_run_caches_only_the_activations_its_plan_reports(small_model, monkeypatch, operation):
    operator, operation_input = operation(small_model)
    planned = set(operator.plan(operation_input, verbose=False).activations)

    runs = record_runs(monkeypatch, small_model)
    operator.apply(operation_input, verbose=False)

    caches = [run.cache for run in runs if run.cache is not None]
    assert caches
    for cache in caches:
        assert planned <= set(cache)
        # A bridge also files an activation under the name of its module's own output hook; such an
        # alias holds the very tensor of a planned name, and nothing else may come along.
        for name in set(cache) - planned:
            assert any(torch.equal(cache[name], cache[alias]) for alias in planned), name


@pytest.mark.parametrize("operation", list(CACHED_PROMPTS), ids=[IDS[operation] for operation in CACHED_PROMPTS])
def test_an_input_runs_its_prompts_once_however_often_it_is_planned_and_run(small_model, monkeypatch, operation):
    # Planning resolves the index without a forward pass, and an input keeps the activations it
    # gathered: a second run of the same input reads them again instead of running the model.
    operator, operation_input = operation(small_model)
    runs = record_runs(monkeypatch, small_model)

    operator.plan(operation_input, verbose=False)
    assert runs == []
    operator.apply(operation_input, verbose=False)
    operator.apply(operation_input, verbose=False)

    assert len([run for run in runs if run.entry == "run_with_cache"]) == CACHED_PROMPTS[operation]


@pytest.mark.parametrize("operation", list(CACHED_PROMPTS), ids=[IDS[operation] for operation in CACHED_PROMPTS])
def test_an_input_moves_every_tensor_it_holds(small_model, operation):
    # ``meta`` exists on every build, so the move is observable on a runner with no accelerator. The
    # input's own attributes are walked rather than its properties: a tensor that a later change adds
    # and ``to`` forgets is what this is here to catch.
    operator, operation_input = operation(small_model)
    operator.apply(operation_input, verbose=False)
    held = _tensors(operation_input)
    assert held and {tensor.device.type for tensor in held} == {"cpu"}

    assert operation_input.to("meta") is operation_input

    moved = _tensors(operation_input)
    assert len(moved) == len(held)
    assert {tensor.device.type for tensor in moved} == {"meta"}


def _tensors(value) -> list[torch.Tensor]:
    """Every tensor reachable from ``value`` through tensors, mappings, activation caches and sequences."""
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict) or hasattr(value, "cache_dict"):
        return [tensor for element in value.values() for tensor in _tensors(element)]
    if isinstance(value, list | tuple):
        return [tensor for element in value for tensor in _tensors(element)]
    if isinstance(value, ProbeInput | LensInput | PatchingInput):
        return [tensor for attribute in vars(value).values() for tensor in _tensors(attribute)]
    return []


# --------------------------------------------------------------------------------------------------
# Reading a config - every field is read strictly
# --------------------------------------------------------------------------------------------------
ELEMENT = {"activation": "resid.post", "location": ["layer"], "filter": [{"layer": 1}], "type": "replace"}
RAW_CONFIGS = {
    LensConfig: {"method": "logit_lens", "top_k": 2, "source": {"activation": "resid.post", "location": ["position"]}},
    PatchingConfig: {
        "metric": {
            "logit_diff": {"type": "logit_diff", "tokens": ["correct", "distractor"], "comparison": "indirect_effect"}
        },
        "method": "exact",
        "patch": [{"source": ELEMENT, "target": ELEMENT}],
    },
    ProbeConfig: {
        "method": {"type": "linear", "max_iter": 50},
        "pool": {"position": "mean"},
        "metric": ["accuracy"],
        "source": {"activation": "resid.post", "location": ["layer"]},
    },
    # An int written for a float field is read as the float it means.
    CircuitConfig: {"method": {"type": "information_flow_routes", "threshold": 0}},
}


def test_an_int_written_for_a_float_field_is_read_as_a_float(config_from_yaml):
    config = config_from_yaml(CircuitConfig, "raw", RAW_CONFIGS[CircuitConfig])

    assert type(config.method.threshold) is float


@pytest.mark.parametrize(
    ("config_class", "raw", "message"),
    [
        (
            CircuitConfig,
            {"method": {"type": "information_flow_routes", "threshold": "0.04"}},
            "Expected a float, got '0.04'",
        ),
        (LensConfig, {**RAW_CONFIGS[LensConfig], "top_k": "2"}, "Expected an int, got '2'"),
        # a bool is an int to Python, but not a count of tokens
        (LensConfig, {**RAW_CONFIGS[LensConfig], "top_k": True}, "Expected an int, got True"),
        (ProbeConfig, {**RAW_CONFIGS[ProbeConfig], "method": {"type": "linear", "max_iter": 1.5}}, "Expected an int"),
        (
            PatchingConfig,
            {**RAW_CONFIGS[PatchingConfig], "patch": [{"source": {**ELEMENT, "value": "zero"}, "target": ELEMENT}]},
            "Expected a float, got 'zero'",
        ),
    ],
)
def test_a_value_its_field_cannot_take_is_rejected_as_the_config_is_read(config_from_yaml, config_class, raw, message):
    with pytest.raises(ValueError, match=message):
        config_from_yaml(config_class, "broken", raw)


def test_an_environment_variable_is_read_as_the_value_it_holds(config_from_yaml, monkeypatch):
    # A value that is a single variable reads as if the variable's value were written in its place.
    monkeypatch.setenv("ENTOSCOPE_TEST_THRESHOLD", "0.5")
    monkeypatch.setenv("ENTOSCOPE_TEST_POSITION", "-2")
    raw = {
        "method": {
            "type": "information_flow_routes",
            "threshold": "${ENTOSCOPE_TEST_THRESHOLD}",
            "prediction_positions": ["${ENTOSCOPE_TEST_POSITION}"],
        }
    }

    config = config_from_yaml(CircuitConfig, "env", raw)

    assert config.method.threshold == 0.5
    assert config.method.prediction_positions == [-2]
