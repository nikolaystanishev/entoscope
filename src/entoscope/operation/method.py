"""The method config a config runs its method with, and how a method names its class."""

import builtins
from dataclasses import dataclass
from typing import Any, TypeVar, get_args

from entoscope.operation.registry import RegistryFactory
from entoscope.util.data import dataclass_from_dict, read_fields


@dataclass(kw_only=True)
class MethodConfig:
    """The base of every method config: the method a config runs, and that method's settings.

    Each operation's method config base extends this one and names its ``factory``. ``type`` has no
    default because one method config class serves every method that takes it (each method with no
    fields of its own takes its operation's base), so only the instance can say which method it
    configures.

    Every field is read as a config file's would be, however the method config is built.

    Attributes:
        type: The name the method is registered under.

    Raises:
        ValueError: If a field holds a value its type cannot take.
    """

    type: str

    def __post_init__(self):
        read_fields(self)

    @classmethod
    def factory(cls) -> builtins.type[RegistryFactory[Any]]:  # ``type`` is the field here
        """Name the factory the operation's methods register with.

        Returns:
            The operation's method factory.

        Raises:
            NotImplementedError: If the class names no factory.
        """
        raise NotImplementedError(f"{cls.__name__} names no factory its methods register with")

    @classmethod
    def from_config(cls, value: Any) -> "MethodConfig":
        """Read a config file's ``method`` into the method config class its method takes.

        Args:
            value: The method's name alone, or a mapping of ``type`` and the method's fields.

        Returns:
            An instance of the config class the method named by ``type`` declares.

        Raises:
            ValueError: If ``value`` is neither a name nor a mapping, names no ``type``, names a method
                nothing registered, or has a key the method config does not.
        """
        if isinstance(value, str):
            value = {"type": value}
        if not isinstance(value, dict):
            raise ValueError(f"Expected a {cls.factory().kind} or a mapping for {cls.__name__}, got {value!r}")
        if "type" not in value:
            raise ValueError(f"Missing required keys for {cls.__name__}: ['type']")
        return dataclass_from_dict(method_config_class(cls.factory().get(value["type"])), value)


def method_config_class(method: type) -> type[MethodConfig]:
    """Find the config class a method or a dataset names as its first type argument.

    A TypeVar left open gives its bound, and a class naming none takes that of the class it extends.
    Type arguments are erased at runtime, so the subscripted base in a class statement is their only
    record.

    Args:
        method: A method or dataset class.

    Returns:
        The config class it is configured with.

    Raises:
        TypeError: If neither the class nor any class it extends names a config class.
    """
    for cls in method.__mro__:
        for base in cls.__dict__.get("__orig_bases__", ()):
            if arguments := get_args(base):
                method_config = arguments[0].__bound__ if isinstance(arguments[0], TypeVar) else arguments[0]
                if method_config is not None:
                    return method_config
    raise TypeError(f"{method.__name__} names no method config")
