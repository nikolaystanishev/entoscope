"""The examples, used the way their README says: every config loads, and every script plans every config."""

import runpy
import sys
from pathlib import Path

import pytest
import yaml

from intex.circuit.config import CircuitConfig
from intex.lens.config import LensConfig
from intex.operation.config import ConfigFactory
from intex.patching.config import PatchingConfig
from intex.probe.config import ProbeConfig

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
