import subprocess
import sys
from dataclasses import dataclass, field

import pytest

from intex.script import experiment

DESCRIPTION = "module level description"

SCRIPT_SOURCE = """\
from dataclasses import dataclass, field

from intex.script import experiment

DESCRIPTION = "Example experiment script"


@dataclass
class Config:
    config_path: str = field(metadata={"help": "data config file path"})
    steps: int = 10
    lr: float = 0.1
    verbose: bool = False
    seed: int | None = None
    mode: str = field(default="fast", metadata={"choices": ["fast", "slow"]})
    tags: list[str] = field(default_factory=list, metadata={"help": "run tags"})
    params: dict[str, int] = field(default_factory=dict, metadata={"help": "extra params"})
    size: tuple[int, int] = (2, 3)
    values: tuple[float, ...] = ()


@experiment
def main(config: Config):
    print(config)
"""


def run_script(tmp_path, *args):
    script = tmp_path / "experiment_script.py"
    script.write_text(SCRIPT_SOURCE)
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True)


@dataclass
class Config:
    name: str = field(metadata={"help": "run name"})
    steps: int = 1


def test_parses_argv_into_config():
    @experiment
    def main(config: Config):
        return config

    assert main(argv=["--name", "x", "--steps", "3"]) == Config(name="x", steps=3)


def test_explicit_config_bypasses_parser():
    @experiment
    def main(config: Config):
        return config

    config = Config(name="x")
    assert main(config) is config


def test_wrapper_preserves_function_name():
    @experiment
    def main(config: Config):
        return config

    assert main.__name__ == "main"


def test_description_argument_wins(capsys):
    @experiment(description="custom description")
    def main(config: Config):
        return config

    with pytest.raises(SystemExit):
        main(argv=["--help"])
    help_text = capsys.readouterr().out
    assert "custom description" in help_text
    assert DESCRIPTION not in help_text


def test_description_falls_back_to_module_description(capsys):
    @experiment
    def main(config: Config):
        return config

    with pytest.raises(SystemExit):
        main(argv=["--help"])
    assert DESCRIPTION in capsys.readouterr().out


def test_description_falls_back_to_docstring(capsys):
    namespace = {"Config": Config}
    exec('def main(config: Config):\n    """docstring description"""\n    return config\n', namespace)
    main = experiment(namespace["main"])

    with pytest.raises(SystemExit):
        main(argv=["--help"])
    assert "docstring description" in capsys.readouterr().out


def test_unannotated_first_parameter_raises():
    with pytest.raises(TypeError, match="annotate its first parameter"):

        @experiment
        def main(config):
            return config


def test_no_parameters_raises():
    with pytest.raises(TypeError, match="annotate its first parameter"):

        @experiment
        def main():
            return None


def test_non_dataclass_annotation_raises():
    with pytest.raises(TypeError, match="not a dataclass"):

        @experiment
        def main(config: int):
            return config


def test_auto_runs_when_defined_in_main_module(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--name", "x"])
    calls = []

    def main(config: Config):
        calls.append(config)

    main.__module__ = "__main__"
    experiment(main)
    assert calls == [Config(name="x")]


def test_does_not_auto_run_outside_main_module():
    calls = []

    @experiment
    def main(config: Config):
        calls.append(config)

    assert calls == []


def test_script_end_to_end(tmp_path):
    # One value for every supported field type: scalars, bool, optional, choices, list, dict, tuples.
    result = run_script(
        tmp_path,
        *("--config-path", "foo.yaml", "--steps", "5", "--lr", "0.5", "--verbose", "--seed", "7"),
        *("--mode", "slow", "--tags", "a", "b", "--params", "a:1", "b:2", "--size", "4", "5"),
        *("--values", "0.5", "1.5"),
    )
    assert result.returncode == 0
    expected = (
        "Config(config_path='foo.yaml', steps=5, lr=0.5, verbose=True, seed=7, mode='slow', "
        "tags=['a', 'b'], params={'a': 1, 'b': 2}, size=(4, 5), values=(0.5, 1.5))"
    )
    assert result.stdout.strip() == expected


def test_script_defaults_end_to_end(tmp_path):
    result = run_script(tmp_path, "--config-path", "foo.yaml")
    assert result.returncode == 0
    expected = (
        "Config(config_path='foo.yaml', steps=10, lr=0.1, verbose=False, seed=None, mode='fast', "
        "tags=[], params={}, size=(2, 3), values=())"
    )
    assert result.stdout.strip() == expected


def test_script_missing_required_argument(tmp_path):
    result = run_script(tmp_path)
    assert result.returncode == 2
    assert "--config-path" in result.stderr
