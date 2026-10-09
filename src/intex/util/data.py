import dataclasses
import json
import os
import re
import types
import typing
from enum import Enum
from typing import Any

import yaml


def write_json(file_path: str, data: dict | list):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def read_yaml(file_path: str) -> list | dict:
    loader = yaml.FullLoader
    tag = "!ENV"
    pattern = re.compile(r".*?\${(\w+)}.*?")
    loader.add_implicit_resolver(tag, pattern, None)

    def constructor_env_variables(yaml_loader: yaml.FullLoader, node):
        value = yaml_loader.construct_scalar(node)
        match = pattern.findall(value)  # to find all env variables in line
        if match:
            full_value = value
            for g in match:
                key = f"${{{g}}}"
                env_value = os.environ.get(g, g)
                if key == full_value and env_value.isdigit():
                    full_value = int(env_value)
                else:
                    full_value = full_value.replace(key, env_value)
            return full_value
        return value

    loader.add_constructor(tag, constructor_env_variables)

    with open(file_path, encoding="utf-8") as f:
        return yaml.load(f, Loader=loader)


def create_if_not_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)


def from_dict(cls: type, value: Any) -> Any:
    """Deserialize a raw YAML/JSON shape into ``cls``, driven by its type hints.

    Values that are already of the target type pass through untouched, so raw shapes and
    code-built objects can mix freely. A ``from_config`` classmethod on the target type
    overrides the default handling (for fields whose config shape differs from their fields);
    enums are constructed by value, and dataclasses recurse field-wise from a mapping —
    strictly, so unknown or missing keys raise ``ValueError``. ``list``/``dict`` annotations
    recurse into their elements (keys and values for dicts), ``dict`` subclasses construct
    through the key/value types of their generic base, ``X | None`` unwraps to ``X``, and
    anything else is returned as-is. The input is never mutated (YAML anchors may alias sub-mappings).
    ``to_dict`` writes each of these shapes back, a ``to_config`` method standing in for ``from_config``.
    """
    if cls is Any:
        return value
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
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"Expected a mapping for {cls.__name__}, got {value!r}")
        key_type, value_type = typing.get_args(cls) or (Any, Any)
        return {from_dict(key_type, key): from_dict(value_type, element) for key, element in value.items()}
    if not isinstance(cls, type) or isinstance(value, cls):
        return value
    if hasattr(cls, "from_config"):
        return cls.from_config(value)
    if issubclass(cls, Enum):
        return cls(value)
    if issubclass(cls, dict):
        if not isinstance(value, dict):
            raise ValueError(f"Expected a mapping for {cls.__name__}, got {value!r}")
        key_type, value_type = next(
            (typing.get_args(base) for base in getattr(cls, "__orig_bases__", ()) if typing.get_origin(base) is dict),
            (Any, Any),
        )
        return cls({from_dict(key_type, key): from_dict(value_type, element) for key, element in value.items()})
    if dataclasses.is_dataclass(cls):
        return _dataclass_from_dict(cls, value)
    return value


def to_dict(value: Any) -> Any:
    """The raw JSON shape of ``value``, which ``from_dict`` reads back."""
    return json.loads(json.dumps(value, default=_raw))


def _raw(value: Any) -> Any:
    return value.to_config() if hasattr(value, "to_config") else vars(value)


def _dataclass_from_dict(cls: type, value: Any) -> Any:
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
