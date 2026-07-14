from dataclasses import dataclass, field

import pytest

from intex.script.argument import build_parser, parse_config


@dataclass
class FullConfig:
    name: str = field(metadata={"help": "run name"})
    steps: int = 10
    lr: float = 0.1
    verbose: bool = False
    seed: int | None = None
    mode: str = field(default="fast", metadata={"choices": ["fast", "slow"]})
    tags: list[str] = field(default_factory=list, metadata={"help": "run tags"})
    params: dict[str, int] = field(default_factory=dict, metadata={"help": "extra params"})
    size: tuple[int, int] = (2, 3)


def test_defaults_applied():
    config = parse_config(FullConfig, ["--name", "x"])
    assert config == FullConfig(name="x", steps=10, lr=0.1, verbose=False, seed=None, mode="fast")


def test_type_conversion():
    config = parse_config(FullConfig, ["--name", "x", "--steps", "5", "--lr", "0.5"])
    assert config.steps == 5
    assert config.lr == 0.5


def test_underscored_field_becomes_dashed_flag():
    @dataclass
    class Config:
        config_path: str = field(metadata={"help": "data config file path"})

    config = parse_config(Config, ["--config-path", "foo.yaml"])
    assert config.config_path == "foo.yaml"


def test_bool_flag():
    assert parse_config(FullConfig, ["--name", "x", "--verbose"]).verbose is True
    assert parse_config(FullConfig, ["--name", "x", "--no-verbose"]).verbose is False


def test_optional_annotation_unwrapped():
    config = parse_config(FullConfig, ["--name", "x", "--seed", "7"])
    assert config.seed == 7


def test_list_field_parses_multiple_values():
    config = parse_config(FullConfig, ["--name", "x", "--tags", "a", "b"])
    assert config.tags == ["a", "b"]


def test_list_field_defaults_to_factory():
    assert parse_config(FullConfig, ["--name", "x"]).tags == []


def test_list_field_converts_element_type():
    @dataclass
    class Config:
        values: list[int] = field(default_factory=list)

    assert parse_config(Config, ["--values", "1", "2"]).values == [1, 2]


def test_metadata_nargs_overrides_derived():
    @dataclass
    class Config:
        values: list[int] = field(default_factory=list, metadata={"nargs": "+"})

    assert parse_config(Config, ["--values", "1", "2"]).values == [1, 2]
    with pytest.raises(SystemExit):
        parse_config(Config, ["--values"])  # "+" requires at least one value


def test_dict_field_parses_key_value_tokens():
    config = parse_config(FullConfig, ["--name", "x", "--params", "a:1", "b:2"])
    assert config.params == {"a": 1, "b": 2}


def test_dict_field_defaults_to_factory():
    assert parse_config(FullConfig, ["--name", "x"]).params == {}


def test_dict_field_converts_key_and_value_types():
    @dataclass
    class Config:
        params: dict[int, float] = field(default_factory=dict)

    assert parse_config(Config, ["--params", "1:0.5"]).params == {1: 0.5}


def test_bare_dict_field_keeps_strings():
    @dataclass
    class Config:
        params: dict = field(default_factory=dict)

    assert parse_config(Config, ["--params", "1:2"]).params == {"1": "2"}


def test_dict_value_may_contain_colons():
    @dataclass
    class Config:
        params: dict = field(default_factory=dict)

    assert parse_config(Config, ["--params", "url:http://x"]).params == {"url": "http://x"}


def test_dict_token_without_separator_exits():
    with pytest.raises(SystemExit) as exc:
        parse_config(FullConfig, ["--name", "x", "--params", "a-1"])
    assert exc.value.code == 2


def test_dict_unconvertible_value_exits():
    with pytest.raises(SystemExit) as exc:
        parse_config(FullConfig, ["--name", "x", "--params", "a:x"])
    assert exc.value.code == 2


def test_dict_repeated_option_keeps_last_and_spares_default():
    @dataclass
    class Config:
        params: dict = field(default_factory=dict)

    parser = build_parser(Config)
    assert vars(parser.parse_args(["--params", "a:1", "--params", "b:2"]))["params"] == {"b": "2"}
    assert vars(parser.parse_args([]))["params"] == {}  # the default dict was not mutated


def test_dict_field_help_shows_metavar_and_default():
    help_text = build_parser(FullConfig).format_help()
    assert "[KEY:VALUE ...]" in help_text
    assert "extra params (default: {})" in help_text


def test_tuple_field_converts_each_position():
    @dataclass
    class Config:
        pair: tuple[str, int] = ("a", 0)

    assert parse_config(Config, ["--pair", "b", "2"]).pair == ("b", 2)


def test_tuple_field_defaults_applied():
    config = parse_config(FullConfig, ["--name", "x"])
    assert config.size == (2, 3)
    assert parse_config(FullConfig, ["--name", "x", "--size", "4", "5"]).size == (4, 5)


def test_tuple_field_wrong_arity_exits():
    with pytest.raises(SystemExit) as exc:
        parse_config(FullConfig, ["--name", "x", "--size", "4"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit):
        parse_config(FullConfig, ["--name", "x", "--size", "4", "5", "6"])


def test_tuple_field_unconvertible_value_exits():
    with pytest.raises(SystemExit) as exc:
        parse_config(FullConfig, ["--name", "x", "--size", "4", "b"])
    assert exc.value.code == 2


def test_homogeneous_tuple_field_takes_any_length():
    @dataclass
    class Config:
        values: tuple[int, ...] = ()

    assert parse_config(Config, ["--values", "1", "2", "3"]).values == (1, 2, 3)
    assert parse_config(Config, []).values == ()


def test_bare_tuple_field_keeps_strings():
    @dataclass
    class Config:
        items: tuple = ()

    assert parse_config(Config, ["--items", "a", "b"]).items == ("a", "b")


def test_tuple_field_help_shows_typed_metavar_and_default():
    help_text = build_parser(FullConfig).format_help()
    assert "--size INT INT" in help_text
    assert "(default: (2, 3))" in help_text


def test_missing_required_field_exits():
    with pytest.raises(SystemExit) as exc:
        parse_config(FullConfig, [])
    assert exc.value.code == 2


def test_metadata_choices_enforced():
    assert parse_config(FullConfig, ["--name", "x", "--mode", "slow"]).mode == "slow"
    with pytest.raises(SystemExit):
        parse_config(FullConfig, ["--name", "x", "--mode", "wrong"])


def test_metadata_help_and_description_in_help_text():
    help_text = build_parser(FullConfig, "parser description").format_help()
    assert "parser description" in help_text
    assert "run name" in help_text


def test_defaults_shown_in_help():
    help_text = build_parser(FullConfig).format_help()
    assert "(default: 10)" in help_text  # steps, which has no metadata help
    assert "(default: fast)" in help_text  # mode
    assert help_text.count("(default: False)") == 1  # verbose; BooleanOptionalAction must not double it


def test_required_field_shows_no_default_in_help():
    assert "run name (default" not in build_parser(FullConfig).format_help()


def test_default_appended_to_metadata_help():
    @dataclass
    class Config:
        steps: int = field(default=10, metadata={"help": "number of steps"})

    assert "number of steps (default: 10)" in build_parser(Config).format_help()


def test_metadata_default_placeholder_controls_placement():
    @dataclass
    class Config:
        steps: int = field(default=10, metadata={"help": "steps [%(default)s]"})

    help_text = build_parser(Config).format_help()
    assert "steps [10]" in help_text
    assert "(default:" not in help_text


def test_metadata_overrides_derived_kwargs():
    @dataclass
    class Config:
        tag: str = field(default="x", metadata={"type": str.upper})

    assert parse_config(Config, ["--tag", "abc"]).tag == "ABC"


def test_default_factory_used_as_default():
    @dataclass
    class Config:
        steps: int = field(default_factory=lambda: 3)

    assert parse_config(Config, []).steps == 3
