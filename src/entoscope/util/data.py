import dataclasses
import json
import numbers
import os
import re
import types
import typing
from enum import Enum
from pathlib import PurePath
from typing import Any

import networkx as nx
import yaml


def write_json(file_path: str, data: dict | list):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def write_graph(graph: nx.Graph, save_dir: str, name: str) -> None:
    """Write a graph as both GraphML, which other tools read, and node-link JSON, which reads back exactly.

    Args:
        graph: The graph.
        save_dir: The directory to write into.
        name: The file name, without its extension.
    """
    nx.write_graphml(graph, os.path.join(save_dir, f"{name}.graphml"))
    write_json(os.path.join(save_dir, f"{name}.json"), nx.node_link_data(graph, edges="links"))


_ENV_TAG = "!ENV"
_ENV_VARIABLE = re.compile(r".*?\${(\w+)}.*?")


class _EnvLoader(yaml.FullLoader):
    """A ``FullLoader`` that also replaces ``${NAME}`` with the environment variable's value.

    A subclass, so registering the resolver leaves ``yaml.FullLoader`` itself as every other reader
    expects it.
    """


def _construct_env_variables(loader: _EnvLoader, node: yaml.Node) -> Any:
    value = loader.construct_scalar(node)
    names = _ENV_VARIABLE.findall(value)
    if len(names) == 1 and value == f"${{{names[0]}}}" and names[0] in os.environ:
        return _read_env_value(os.environ[names[0]])
    for name in names:
        value = value.replace(f"${{{name}}}", os.environ.get(name, name))
    return value


def _read_env_value(env_value: str) -> Any:
    """Read a variable's value as if it were written into the file in place of the variable.

    An environment holds only strings, so this is what gives a number, a boolean or ``null`` its type.
    A value that reads as a list or a mapping, or does not parse, is kept as the string it is.

    Args:
        env_value: The variable's value.

    Returns:
        The value as a YAML scalar.
    """
    try:
        scalar = yaml.safe_load(env_value)
    except yaml.YAMLError:
        return env_value
    return scalar if isinstance(scalar, str | int | float | bool) or scalar is None else env_value


_EnvLoader.add_implicit_resolver(_ENV_TAG, _ENV_VARIABLE, None)
_EnvLoader.add_constructor(_ENV_TAG, _construct_env_variables)


def read_yaml(file_path: str) -> list | dict:
    """Read a YAML file, replacing every ``${NAME}`` with the environment variable's value.

    An unset variable is replaced by its own name. A value that is a single variable reads as the
    variable's value would written in its place, so a number, a boolean or ``null`` keeps its type.

    Args:
        file_path: The file to read.

    Returns:
        The file's content.
    """
    with open(file_path, encoding="utf-8") as f:
        return yaml.load(f, Loader=_EnvLoader)


def create_if_not_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)


def from_dict(cls: type, value: Any) -> Any:
    """Read a raw YAML or JSON shape into ``cls``, driven by its type hints.

    Values that are already of the target type pass through untouched, so raw shapes and code-built
    objects can mix freely. A ``from_config`` classmethod on the target type overrides the default
    handling, for a type whose config shape differs from its fields. Enums are constructed by value,
    paths from their string, and dataclasses field by field from a mapping, strictly. ``list``,
    ``tuple``, ``set`` and ``dict`` annotations recurse into their elements (keys and values for
    dicts), ``dict`` subclasses construct through the key and value types of their generic base,
    ``X | None`` unwraps to ``X``, and anything else is returned as-is. ``int``, ``float``, ``str`` and
    ``bool`` take only a value of their own kind, an int reading into a float field as a float and a
    bool counting as no number. The input is never mutated, since YAML anchors may alias
    sub-mappings. ``to_dict`` writes each of these shapes back, a ``to_config`` method standing in for
    ``from_config``.

    Args:
        cls: The type to read into.
        value: The raw shape.

    Returns:
        ``value`` as an instance of ``cls``.

    Raises:
        ValueError: If ``value`` does not have the shape ``cls`` needs, or a dataclass's mapping has an
            unknown key or misses a required one.
    """
    if cls is Any:
        return value
    if cls in _SCALARS:
        return _scalar_from_dict(cls, value)
    origin = typing.get_origin(cls)
    if origin in {typing.Union, types.UnionType}:
        args = [arg for arg in typing.get_args(cls) if arg is not type(None)]
        if value is None or len(args) != 1:
            return value
        return from_dict(args[0], value)
    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"Expected a list for {cls.__name__}, got {value!r}")
        (element_type,) = typing.get_args(cls) or (Any,)
        return [from_dict(element_type, element) for element in value]
    if (origin or cls) is tuple:
        return _tuple_from_dict(cls, value)
    if (origin or cls) in {set, frozenset}:
        if not isinstance(value, list | tuple | set | frozenset):
            raise ValueError(f"Expected a list for {cls}, got {value!r}")
        (element_type,) = typing.get_args(cls) or (Any,)
        return (origin or cls)(from_dict(element_type, element) for element in value)
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"Expected a mapping for {cls.__name__}, got {value!r}")
        key_type, value_type = typing.get_args(cls) or (Any, Any)
        return {_key_from_dict(key_type, key): from_dict(value_type, element) for key, element in value.items()}
    if origin is not None or not isinstance(cls, type) or isinstance(value, cls):
        return value
    if hasattr(cls, "from_config"):
        return cls.from_config(value)
    if issubclass(cls, Enum):
        return cls(value)
    if issubclass(cls, PurePath):
        return cls(value)
    if issubclass(cls, dict):
        if not isinstance(value, dict):
            raise ValueError(f"Expected a mapping for {cls.__name__}, got {value!r}")
        key_type, value_type = next(
            (typing.get_args(base) for base in getattr(cls, "__orig_bases__", ()) if typing.get_origin(base) is dict),
            (Any, Any),
        )
        return cls({_key_from_dict(key_type, key): from_dict(value_type, element) for key, element in value.items()})
    if dataclasses.is_dataclass(cls):
        return dataclass_from_dict(cls, value)
    return value


def to_dict(value: Any) -> Any:
    """Write a value as its raw JSON shape, which ``from_dict`` reads back.

    Args:
        value: A config, or any value a config holds.

    Returns:
        The value as JSON types: dicts, lists, strings, numbers, booleans and ``None``.

    Raises:
        TypeError: If the value, or anything in it, has no such shape, such as a function or a tensor. A
            config holds only data.
    """
    return json.loads(json.dumps(value, default=_raw))


def _raw(value: Any) -> Any:
    if hasattr(value, "to_config"):
        return value.to_config()
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in dataclasses.fields(value) if field.init}
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, set | frozenset):
        return sorted(value, key=repr)
    raise TypeError(
        f"A {type(value).__name__} has no config form: a config holds only numbers, strings, enums and paths, "
        "and lists, tuples, sets, mappings and dataclasses of them"
    )


_SCALARS = {
    bool: ("a bool", lambda value: isinstance(value, bool)),
    int: ("an int", lambda value: isinstance(value, numbers.Integral) and not isinstance(value, bool)),
    float: ("a float", lambda value: isinstance(value, numbers.Real) and not isinstance(value, bool)),
    str: ("a string", lambda value: isinstance(value, str)),
}


def _scalar_from_dict(cls: type, value: Any) -> Any:
    kind, fits = _SCALARS[cls]
    if not fits(value):
        raise ValueError(f"Expected {kind}, got {value!r}")
    return value if type(value) is cls else cls(value)


def _tuple_from_dict(cls: Any, value: Any) -> tuple:
    if not isinstance(value, list | tuple):
        raise ValueError(f"Expected a list for {cls}, got {value!r}")
    element_types = typing.get_args(cls) or (Any, ...)
    if len(element_types) == 2 and element_types[1] is Ellipsis:
        element_types = (element_types[0],) * len(value)
    elif len(element_types) != len(value):
        raise ValueError(f"Expected {len(element_types)} values for {cls}, got {value!r}")
    return tuple(from_dict(element_type, element) for element_type, element in zip(element_types, value, strict=True))


def _key_from_dict(key_type: Any, key: Any) -> Any:
    if key_type in (int, float) and isinstance(key, str):
        return key_type(key)
    return from_dict(key_type, key)


def dataclass_from_dict(cls: type, value: Any) -> Any:
    """Read a mapping field by field into a dataclass, bypassing any ``from_config`` of its own.

    What a ``from_config`` that only picks the class to read into hands the reading on to, since calling
    ``from_dict`` there would call that ``from_config`` again.

    Args:
        cls: The dataclass.
        value: A mapping of its init fields.

    Returns:
        The dataclass instance.

    Raises:
        ValueError: If ``value`` is not a mapping, has a key the dataclass does not, or misses a required
            one.
    """
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping for {cls.__name__}, got {value!r}")
    fields = {field.name: field for field in dataclasses.fields(cls) if field.init}
    unknown = sorted(set(value) - set(fields))
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {unknown}")
    missing = sorted(
        name
        for name, field in fields.items()
        if name not in value and field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING
    )
    if missing:
        raise ValueError(f"Missing required keys for {cls.__name__}: {missing}")
    hints = typing.get_type_hints(cls)
    return cls(**{name: from_dict(hints[name], element) for name, element in value.items()})
