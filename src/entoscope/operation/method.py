"""The method config a config runs its method with, and how a method names its class."""

import builtins
from dataclasses import dataclass
from typing import Any, TypeVar, get_args

from entoscope.operation.registry import RegistryFactory
from entoscope.util.data import dataclass_from_dict


@dataclass(kw_only=True)
class MethodConfig:
    """``type`` names the method; each operation's method config base extends this one and names its ``factory``.

    ``type`` has no default: one method config class serves every method that takes it - each method with no
    fields of its own takes its operation's base - so only the instance can say which method it configures.
    """

    type: str

    @classmethod
    def factory(cls) -> builtins.type[RegistryFactory[Any]]:  # ``type`` is the field here
        """The factory the operation's methods register with."""
        raise NotImplementedError(f"{cls.__name__} names no factory its methods register with")

    @classmethod
    def from_config(cls, value: Any) -> "MethodConfig":
        """A config file's ``method``: a mapping read into the class its ``type``'s method takes, or just the type."""
        if isinstance(value, str):
            value = {"type": value}
        if not isinstance(value, dict):
            raise ValueError(f"Expected a {cls.factory().kind} or a mapping for {cls.__name__}, got {value!r}")
        if "type" not in value:
            raise ValueError(f"Missing required keys for {cls.__name__}: ['type']")
        return dataclass_from_dict(method_config_class(cls.factory().get(value["type"])), value)


def method_config_class(method: type) -> type[MethodConfig]:
    """The config class ``method``, a method or a dataset, names as its first type argument.

    A TypeVar left open gives its bound, and a class naming none takes that of the class it extends.

    Type arguments are erased at runtime: the subscripted base in a class statement is their only record.
    """
    for cls in method.__mro__:
        for base in cls.__dict__.get("__orig_bases__", ()):
            if arguments := get_args(base):
                method_config = arguments[0].__bound__ if isinstance(arguments[0], TypeVar) else arguments[0]
                if method_config is not None:
                    return method_config
    raise TypeError(f"{method.__name__} names no method config")
