"""The examples, used the way their README says: every config loads, and every script plans and runs it."""

import json
import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from entoscope.circuit.config import CircuitConfig
from entoscope.lens.config import LensConfig
from entoscope.operation.config import ConfigFactory
from entoscope.patching.config import PatchingConfig
from entoscope.probe.config import ProbeConfig

from ..conftest import TransformerBridge, forbid_model_runs

EXAMPLES = Path(__file__).parents[2] / "examples"
# Each script runs the entries of the YAML file beside it, which are read into this config class.
SCRIPTS = {
    "lens.py": LensConfig,
    "probe.py": ProbeConfig,
    "patching.py": PatchingConfig,
    "circuit.py": CircuitConfig,
    "custom_lens.py": LensConfig,
}
# Configs naming a method only its script registers.
REGISTERING = {"custom_lens.py"}
# What a run of each script leaves in runs/<config-id>/ beside its config.json.
OUTPUTS = {
    "lens.py": ["result.csv", "result.html"],
    "probe.py": ["result.csv", "probes.pkl"],
    "patching.py": ["result.csv", "result.html"],
    "circuit.py": ["graph_0.graphml", "routes_0.graphml", "circuit_0.html"],
    "custom_lens.py": ["result.csv", "result.html"],
}

ENTRIES = [
    (script, config_id)
    for script in SCRIPTS
    for config_id in yaml.safe_load((EXAMPLES / script).with_suffix(".yaml").read_text(encoding="utf-8"))
]


@pytest.mark.parametrize(("script", "config_id"), ENTRIES)
def test_every_config_loads(script, config_id, registries):
    if script in REGISTERING:
        if TransformerBridge is None:
            pytest.skip("the script registering the method imports TransformerBridge")
        # Run under a name of its own, the module registers its method without running main.
        runpy.run_path(str(EXAMPLES / script), run_name="registering_example")

    config = ConfigFactory[SCRIPTS[script]](str((EXAMPLES / script).with_suffix(".yaml"))).create(config_id)

    assert config.config_id == config_id


@pytest.mark.parametrize(("script", "config_id"), ENTRIES)
def test_every_script_plans_every_config(script, config_id, backend, request, registries, monkeypatch, tmp_path):
    if backend != "bridge":
        pytest.skip("the examples load their model as a TransformerBridge")
    model = request.getfixturevalue("model")
    forbid_model_runs(monkeypatch, model)
    monkeypatch.setattr(TransformerBridge, "boot_transformers", staticmethod(lambda model_name, **kwargs: model))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [script, "--config-id", config_id, "--dry-run"])

    runpy.run_path(str(EXAMPLES / script), run_name="__main__")

    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(("script", "config_id"), ENTRIES)
def test_every_script_runs_every_config(script, config_id, backend, request, registries, monkeypatch, tmp_path):
    if backend != "bridge":
        pytest.skip("the examples load their model as a TransformerBridge")
    model = request.getfixturevalue("model")
    monkeypatch.setattr(TransformerBridge, "boot_transformers", staticmethod(lambda model_name, **kwargs: model))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [script, "--config-id", config_id])

    runpy.run_path(str(EXAMPLES / script), run_name="__main__")

    run_dir = tmp_path / "runs" / config_id
    assert json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["config_id"] == config_id
    for name in OUTPUTS[script]:
        assert (run_dir / name).stat().st_size > 0, name
    if (run_dir / "result.csv").exists():
        assert len(pd.read_csv(run_dir / "result.csv")) > 0
