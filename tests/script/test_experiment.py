import runpy
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from entoscope.script import experiment

DESCRIPTION = "module level description"

SCRIPT = """\
from dataclasses import dataclass, field

from entoscope.script import experiment

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
    counts: list[int] = field(default_factory=list)
    required_counts: list[int] = field(default_factory=list, metadata={"nargs": "+"})
    params: dict[str, int] = field(default_factory=dict, metadata={"help": "extra params"})
    raw_params: dict = field(default_factory=dict)
    ratios: dict[int, float] = field(default_factory=dict)
    size: tuple[int, int] = (2, 3)
    values: tuple[float, ...] = ()
    items: tuple = ()
    tag: str = field(default="x", metadata={"type": str.upper})
    retries: int = field(default_factory=lambda: 3)
    documented: int = field(default=10, metadata={"help": "number of documented steps"})
    placeholder: int = field(default=7, metadata={"help": "placeholder steps [%(default)s]"})


@experiment
def main(config: Config):
    print(config)
"""

# tag is 'X' rather than 'x': argparse runs a string default through the field's own ``type``, which
# metadata set to ``str.upper`` here.
DEFAULTS = (
    "Config(config_path='foo.yaml', steps=10, lr=0.1, verbose=False, seed=None, mode='fast', tags=[], counts=[], "
    "required_counts=[], params={}, raw_params={}, ratios={}, size=(2, 3), values=(), items=(), tag='X', retries=3, "
    "documented=10, placeholder=7)"
)


def write_script(tmp_path, source: str = SCRIPT) -> Path:
    script = tmp_path / "experiment_script.py"
    script.write_text(source)
    return script


def run_script(tmp_path, *args, source: str = SCRIPT) -> subprocess.CompletedProcess:
    script = write_script(tmp_path, source)
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True)


# --------------------------------------------------------------------------------------------------
# A script run from the command line
# --------------------------------------------------------------------------------------------------
def test_a_script_parses_every_field_type(tmp_path):
    result = run_script(
        tmp_path,
        *("--config-path", "foo.yaml", "--steps", "5", "--lr", "0.5", "--verbose", "--seed", "7"),
        *("--mode", "slow", "--tags", "a", "b", "--counts", "1", "2", "--required-counts", "3"),
        *("--params", "a:1", "b:2", "--raw-params", "url:http://x", "--ratios", "1:0.5"),
        *("--size", "4", "5", "--values", "0.5", "1.5", "--items", "a", "b"),
        *("--tag", "abc", "--retries", "9", "--documented", "11", "--placeholder", "12"),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        "Config(config_path='foo.yaml', steps=5, lr=0.5, verbose=True, seed=7, mode='slow', tags=['a', 'b'], "
        "counts=[1, 2], required_counts=[3], params={'a': 1, 'b': 2}, raw_params={'url': 'http://x'}, "
        "ratios={1: 0.5}, size=(4, 5), values=(0.5, 1.5), items=('a', 'b'), tag='ABC', retries=9, documented=11, "
        "placeholder=12)"
    )


def test_a_script_falls_back_to_every_default(tmp_path):
    result = run_script(tmp_path, "--config-path", "foo.yaml")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == DEFAULTS


def test_the_help_text_describes_every_field(tmp_path):
    help_text = run_script(tmp_path, "--help").stdout

    assert "Example experiment script" in help_text  # the module-level DESCRIPTION
    assert "run tags" in help_text  # metadata help
    assert "--config-path" in help_text  # an underscore becomes a dash
    assert "data config file path (default" not in help_text  # a required field has no default to show
    assert "(default: 10)" in help_text  # steps, which has no metadata help
    assert "(default: fast)" in help_text  # mode
    assert help_text.count("(default: False)") == 1  # BooleanOptionalAction must not double it
    assert "[KEY:VALUE ...]" in help_text  # a dict field's metavar
    assert "--size INT INT" in help_text  # a tuple field's typed metavar
    assert "(default: (2, 3))" in help_text
    assert "number of documented steps (default: 10)" in help_text
    assert "placeholder steps [7]" in help_text  # %(default)s in the help controls its placement
    assert "placeholder steps [7] (default" not in help_text


def test_a_script_that_cannot_parse_its_arguments_exits_with_two(tmp_path):
    # The per-option failures are checked in process below; this is the process-level contract.
    result = run_script(tmp_path)

    assert result.returncode == 2
    assert "--config-path" in result.stderr


def test_a_script_run_as_main_runs_itself(tmp_path, monkeypatch, capsys):
    # The same file, executed in this process under the name a script has. monkeypatch sets the
    # process's own argv, which is what a shell does when it invokes the script.
    script = write_script(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(script), "--config-path", "foo.yaml"])

    runpy.run_path(str(script), run_name="__main__")

    # The decorator ran main at decoration time, with the config argparse built from that argv.
    assert capsys.readouterr().out.strip() == DEFAULTS


def test_a_script_that_is_only_imported_does_not_run_itself(tmp_path, monkeypatch, capsys):
    script = write_script(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(script), "--config-path", "foo.yaml"])

    namespace = runpy.run_path(str(script), run_name="experiment_script")

    assert capsys.readouterr().out == ""
    assert callable(namespace["main"])  # still there to be called with a config of one's own


# --------------------------------------------------------------------------------------------------
# Every field type, parsed by the decorated function itself
# --------------------------------------------------------------------------------------------------
@dataclass
class FullConfig:
    """The in-process twin of the script's config: one field per supported type."""

    config_path: str = field(metadata={"help": "data config file path"})
    steps: int = 10
    lr: float = 0.1
    verbose: bool = False
    seed: int | None = None
    mode: str = field(default="fast", metadata={"choices": ["fast", "slow"]})
    tags: list[str] = field(default_factory=list, metadata={"help": "run tags"})
    counts: list[int] = field(default_factory=list)
    required_counts: list[int] = field(default_factory=list, metadata={"nargs": "+"})
    params: dict[str, int] = field(default_factory=dict, metadata={"help": "extra params"})
    raw_params: dict = field(default_factory=dict)
    ratios: dict[int, float] = field(default_factory=dict)
    size: tuple[int, int] = (2, 3)
    values: tuple[float, ...] = ()
    items: tuple = ()
    tag: str = field(default="x", metadata={"type": str.upper})
    retries: int = field(default_factory=lambda: 3)
    run_id: str = field(init=False, default="local")  # not a parser option at all


def parse_argv(config_cls: type, argv: list[str]):
    """The config a decorated main builds from ``argv`` - the wrapper's own API."""

    @experiment
    def main(config: config_cls):
        return config

    return main(argv=argv)


def test_every_field_type_is_parsed_into_the_config():
    config = parse_argv(
        FullConfig,
        [
            *("--config-path", "foo.yaml", "--steps", "5", "--lr", "0.5", "--verbose", "--seed", "7"),
            *("--mode", "slow", "--tags", "a", "b", "--counts", "1", "2", "--required-counts", "3"),
            *("--params", "a:1", "b:2", "--raw-params", "url:http://x", "--ratios", "1:0.5"),
            *("--size", "4", "5", "--values", "0.5", "1.5", "--items", "a", "b"),
            *("--tag", "abc", "--retries", "9"),
        ],
    )

    assert config == FullConfig(
        config_path="foo.yaml",
        steps=5,
        lr=0.5,
        verbose=True,
        seed=7,
        mode="slow",
        tags=["a", "b"],
        counts=[1, 2],
        required_counts=[3],
        params={"a": 1, "b": 2},
        raw_params={"url": "http://x"},
        ratios={1: 0.5},
        size=(4, 5),
        values=(0.5, 1.5),
        items=("a", "b"),
        tag="ABC",
        retries=9,
    )


def test_the_defaults_of_every_field_type_are_applied():
    config = parse_argv(FullConfig, ["--config-path", "foo.yaml"])

    # tag is 'X': argparse runs a string default through the field's own type (str.upper here), and
    # retries comes from its default_factory. run_id has no option, so it keeps its own default.
    assert config == FullConfig(config_path="foo.yaml", tag="X")
    assert (config.tag, config.retries, config.run_id) == ("X", 3, "local")
    assert (config.tags, config.params, config.values, config.items) == ([], {}, (), ())


# --------------------------------------------------------------------------------------------------
# dict fields - key:value tokens collected by DictAction
# --------------------------------------------------------------------------------------------------
@dataclass
class DictConfig:
    params: dict = field(default_factory=dict)
    typed: dict[int, float] = field(default_factory=dict)
    optional: dict = field(default_factory=lambda: {"kept": "default"}, metadata={"nargs": "?"})


def test_a_dict_option_converts_each_side_of_the_token():
    assert parse_argv(DictConfig, ["--typed", "1:0.5", "2:1.5"]).typed == {1: 0.5, 2: 1.5}


def test_a_bare_dict_option_keeps_both_sides_as_strings():
    assert parse_argv(DictConfig, ["--params", "1:2"]).params == {"1": "2"}


def test_a_dict_value_may_contain_colons():
    # The token splits on the first colon only.
    assert parse_argv(DictConfig, ["--params", "url:http://x"]).params == {"url": "http://x"}


def test_a_repeated_dict_option_keeps_the_last_one_and_spares_the_default():
    assert parse_argv(DictConfig, ["--params", "a:1", "--params", "b:2"]).params == {"b": "2"}
    # Each occurrence builds a fresh dict, so the default one was never written into.
    assert parse_argv(DictConfig, []).params == {}


def test_a_dict_option_given_without_a_value_leaves_the_default_alone():
    # metadata nargs="?" lets the option stand on its own; argparse then hands the action no values
    # at all, and the field keeps what it started with.
    assert parse_argv(DictConfig, ["--optional"]).optional == {"kept": "default"}
    assert parse_argv(DictConfig, []).optional == {"kept": "default"}


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--params", "a-1"], "expected KEY:VALUE"),  # no separator
        (["--typed", "a:1"], "invalid KEY:VALUE"),  # the key does not convert
        (["--typed", "1:x"], "invalid KEY:VALUE"),  # the value does not convert
    ],
)
def test_a_dict_option_that_cannot_be_read_exits(capsys, argv, message):
    with pytest.raises(SystemExit) as exit_info:
        parse_argv(DictConfig, argv)

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


# --------------------------------------------------------------------------------------------------
# tuple fields - values collected per position by TupleAction
# --------------------------------------------------------------------------------------------------
@dataclass
class TupleConfig:
    pair: tuple[str, int] = ("a", 0)
    values: tuple[int, ...] = ()
    items: tuple = ()
    # Homogeneous, because a fixed-arity tuple derives a metavar per position that nargs="?" rejects.
    optional: tuple[int, ...] = field(default=(1, 2), metadata={"nargs": "?"})


def test_a_tuple_option_converts_each_position_with_its_own_type():
    assert parse_argv(TupleConfig, ["--pair", "b", "2"]).pair == ("b", 2)


def test_a_homogeneous_tuple_option_takes_any_number_of_values():
    assert parse_argv(TupleConfig, ["--values", "1", "2", "3"]).values == (1, 2, 3)
    assert parse_argv(TupleConfig, ["--values"]).values == ()
    assert parse_argv(TupleConfig, []).values == ()


def test_a_bare_tuple_option_keeps_its_values_as_strings():
    assert parse_argv(TupleConfig, ["--items", "a", "b"]).items == ("a", "b")


def test_a_tuple_option_given_without_a_value_leaves_the_default_alone():
    # As for a dict: with nargs="?" the option may stand on its own, and the default survives.
    assert parse_argv(TupleConfig, ["--optional"]).optional == (1, 2)
    assert parse_argv(TupleConfig, []).optional == (1, 2)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--pair", "b"], "expected 2 arguments"),  # a fixed tuple takes exactly its own arity
        (["--pair", "b", "2", "3"], "unrecognized arguments"),
        (["--pair", "b", "x"], "invalid value"),  # the second position does not convert
        (["--values", "1", "x"], "invalid value"),
    ],
)
def test_a_tuple_option_that_cannot_be_read_exits(capsys, argv, message):
    with pytest.raises(SystemExit) as exit_info:
        parse_argv(TupleConfig, argv)

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


# --------------------------------------------------------------------------------------------------
# list and bool fields
# --------------------------------------------------------------------------------------------------
@dataclass
class ListConfig:
    names: list = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    required: list[int] = field(default_factory=list, metadata={"nargs": "+"})


def test_a_list_option_converts_every_element():
    assert parse_argv(ListConfig, ["--counts", "1", "2"]).counts == [1, 2]


def test_a_bare_list_option_keeps_its_elements_as_strings():
    assert parse_argv(ListConfig, ["--names", "a", "b"]).names == ["a", "b"]


def test_a_list_option_is_empty_unless_it_is_given():
    assert parse_argv(ListConfig, []).counts == []
    assert parse_argv(ListConfig, ["--counts"]).counts == []


def test_metadata_nargs_overrides_the_derived_one(capsys):
    # "+" replaces the "*" a list field derives, so the option needs at least one value.
    assert parse_argv(ListConfig, ["--required", "3"]).required == [3]

    with pytest.raises(SystemExit):
        parse_argv(ListConfig, ["--required"])
    assert "expected at least one argument" in capsys.readouterr().err


def test_a_boolean_field_takes_both_flags():
    assert parse_argv(FullConfig, ["--config-path", "foo.yaml", "--verbose"]).verbose is True
    assert parse_argv(FullConfig, ["--config-path", "foo.yaml", "--no-verbose"]).verbose is False


def test_a_value_outside_the_metadata_choices_exits(capsys):
    with pytest.raises(SystemExit):
        parse_argv(FullConfig, ["--config-path", "foo.yaml", "--mode", "wrong"])

    assert "invalid choice" in capsys.readouterr().err


def test_a_value_that_does_not_convert_exits(capsys):
    with pytest.raises(SystemExit):
        parse_argv(FullConfig, ["--config-path", "foo.yaml", "--steps", "x"])

    assert "invalid int value" in capsys.readouterr().err


# --------------------------------------------------------------------------------------------------
# The decorated function called directly
# --------------------------------------------------------------------------------------------------
@dataclass
class Config:
    name: str = field(metadata={"help": "run name"})
    steps: int = 1


def test_the_wrapper_executes_without_config():
    @experiment
    def main():
        return "pass"

    assert main() == "pass"


def test_the_wrapper_parses_the_argv_it_is_given():
    @experiment
    def main(config: Config):
        return config

    assert main(argv=["--name", "x", "--steps", "3"]) == Config(name="x", steps=3)
    assert main.__name__ == "main"


def test_an_explicit_config_bypasses_the_parser():
    @experiment
    def main(config: Config):
        return config

    config = Config(name="x")
    assert main(config) is config


def test_the_description_argument_wins(capsys):
    @experiment(description="custom description")
    def main(config: Config):
        return config

    with pytest.raises(SystemExit):
        main(argv=["--help"])

    help_text = capsys.readouterr().out
    assert "custom description" in help_text
    assert DESCRIPTION not in help_text


def test_the_description_falls_back_to_the_modules(capsys):
    @experiment
    def main(config: Config):
        return config

    with pytest.raises(SystemExit):
        main(argv=["--help"])

    assert DESCRIPTION in capsys.readouterr().out


def test_the_description_falls_back_to_the_docstring(capsys):
    # Defined without the module's DESCRIPTION in scope, so the docstring is all that is left.
    namespace = {"Config": Config}
    exec('def main(config: Config):\n    """docstring description"""\n    return config\n', namespace)
    main = experiment(namespace["main"])

    with pytest.raises(SystemExit):
        main(argv=["--help"])

    assert "docstring description" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("def main(config):\n    return config\n", "annotate its first parameter"),
        ("def main(config: int):\n    return config\n", "not a dataclass"),
        ("def main(*, config: Config):\n    return config\n", "must accept a positional argument"),
        ("def main(config: Config, steps):\n    return config\n", "additional required parameters"),
    ],
)
def test_a_function_the_decorator_cannot_build_a_parser_for_raises(source, message):
    namespace = {"Config": Config}
    exec(source, namespace)

    with pytest.raises(TypeError, match=message):
        experiment(namespace["main"])
