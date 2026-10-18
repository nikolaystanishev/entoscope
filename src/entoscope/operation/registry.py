"""Name-to-class registries: the extension points a user fills, a dataset or an operation's method.

The built-ins register through the same ``register`` a user calls, so a name entoscope ships with differs
from a user's only in having been taken first.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, ClassVar, Generic, TypeVar, get_args, get_origin

T = TypeVar("T")
ImplementationT = TypeVar("ImplementationT", bound=type)


def _same_definition(one: type, other: type) -> bool:
    # Re-running the statement that defines a class - a notebook cell, an ``importlib.reload`` -
    # builds a new class object under the same module and qualified name. A class defined in a
    # function is built anew by every call, each a different implementation, so it never counts.
    return one is other or (
        one.__module__ == other.__module__
        and one.__qualname__ == other.__qualname__
        and "<locals>" not in one.__qualname__
    )


class RegistryFactory(Generic[T]):
    """Resolves a name to the class registered under it, which extends the class the factory names.

    A subclass names that class as its type argument and what an implementation is called in messages as
    ``kind``, and each keeps its own entries. A name belongs to one implementation, so a typo or a second
    implementation cannot shadow the first one, while defining the same class again replaces it, which is
    what re-running a notebook cell does.

    Attributes:
        kind: What an implementation is called in messages, such as ``"lens method"``.
    """

    kind: ClassVar[str]

    _base: ClassVar[type]
    _registry: ClassVar[dict[str, type]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        arguments = {get_origin(alias): get_args(alias) for alias in cls.__dict__.get("__orig_bases__", ())}
        base = arguments.get(RegistryFactory, (None,))[0]
        if not isinstance(base, type):
            raise TypeError(f"{cls.__name__} must name the class it registers, as RegistryFactory[TheBaseClass]")
        if not isinstance(getattr(cls, "kind", None), str):
            raise TypeError(f"{cls.__name__} must name what it registers in 'kind'")

        cls._base = base
        cls._registry = {}

    @classmethod
    def register(cls, name: str) -> Callable[[ImplementationT], ImplementationT]:
        """Register the decorated class under ``name``.

        The class is checked as it is registered rather than at its first run.

        Args:
            name: The name a config's ``type`` selects the class by.

        Returns:
            A class decorator that registers the class and returns it unchanged.

        Raises:
            ValueError: If ``name`` is empty or already taken by another class.
            TypeError: If the decorated class does not extend the factory's base class, or leaves one of
                its abstract methods unimplemented.
        """
        key = cls._key(name)

        def decorator(implementation: ImplementationT) -> ImplementationT:
            cls._admit(key, implementation)
            return implementation

        return decorator

    @classmethod
    def get(cls, name: str) -> type[T]:
        """Look up the class registered under ``name``.

        Args:
            name: A registered name.

        Returns:
            The class registered under ``name``.

        Raises:
            ValueError: If nothing is registered under ``name``. The message lists the registered names.
        """
        key = cls._key(name)
        if key not in cls._registry:
            raise ValueError(
                f"Unknown {cls.kind}: {key}. Registered {cls.kind}s: {cls.registered()}. "
                "An implementation is registered when the module defining it is imported"
            )
        return cls._registry[key]

    @classmethod
    def registered(cls) -> list[str]:
        """List the registered names.

        Returns:
            Every name registered with this factory, sorted.
        """
        return sorted(cls._registry)

    @classmethod
    def _key(cls, name: str) -> str:
        if not isinstance(name, str) or not name:
            raise ValueError(f"A {cls.kind} is named by a non-empty string, got {name!r}")
        return name

    @classmethod
    def _admit(cls, key: str, implementation: type) -> None:
        """Store ``implementation`` under ``key``, checked now rather than at its first run.

        Args:
            key: The name to register under.
            implementation: The class to register.

        Raises:
            TypeError: If ``implementation`` does not extend the base class or is abstract.
            ValueError: If ``key`` belongs to another class already.
        """
        if not (isinstance(implementation, type) and issubclass(implementation, cls._base)):
            raise TypeError(f"A {cls.kind} must subclass {cls._base.__name__}, got {implementation!r}")
        if inspect.isabstract(implementation):
            missing = ", ".join(sorted(implementation.__abstractmethods__))
            raise TypeError(
                f"{implementation.__name__} cannot be registered as {cls.kind} '{key}': it does not implement {missing}"
            )
        registered = cls._registry.get(key)
        if registered is not None and not _same_definition(registered, implementation):
            raise ValueError(
                f"{cls.kind.capitalize()} '{key}' is already registered to {registered.__name__}; "
                f"{implementation.__name__} must use another name"
            )
        cls._registry[key] = implementation
