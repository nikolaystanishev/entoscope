import dataclasses
import functools
import inspect
import typing
from collections.abc import Callable, Sequence

from intex.script.argument import parse_config


def _config_class(func: Callable) -> type | None:
    params = list(inspect.signature(func).parameters.values())
    if not params:
        return None
    hints = typing.get_type_hints(func)
    if params[0].name not in hints:
        raise TypeError(f"{func.__name__} must annotate its first parameter with a config dataclass")
    if params[0].kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
        raise TypeError(f"{func.__name__} config parameter must accept a positional argument")
    required = [
        param.name
        for param in params[1:]
        if param.default is inspect.Parameter.empty
        and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if required:
        raise TypeError(f"{func.__name__} has additional required parameters: {required}")
    config_cls = hints[params[0].name]
    if not dataclasses.is_dataclass(config_cls):
        raise TypeError(f"{func.__name__} first parameter annotation {config_cls!r} is not a dataclass")
    return config_cls


def experiment(func: Callable | None = None, *, description: str | None = None) -> Callable:
    """Turn ``main(config: Config)`` into a CLI entry point.

    The config dataclass is taken from the first parameter's annotation and its fields become
    ``--field-name`` options (see ``intex.script.argument.build_parser``). The parser description
    is ``description`` if given, else a module-level ``DESCRIPTION``, else the function docstring.

    When the decorated function lives in the ``__main__`` module it runs immediately at decoration
    time, so the config class and ``DESCRIPTION`` must be defined above it in the script. The
    returned wrapper stays callable with an explicit config (``main(Config(...))``) or a custom
    argv (``main(argv=[...])``), which bypasses/feeds the parser respectively.
    """

    def wrap(fn: Callable) -> Callable:
        config_cls = _config_class(fn)
        desc = description if description is not None else fn.__globals__.get("DESCRIPTION", fn.__doc__)

        @functools.wraps(fn)
        def wrapper(config=None, argv: Sequence[str] | None = None):
            if config_cls is None:
                return fn()
            if config is None:
                config = parse_config(config_cls, argv, desc)
            return fn(config)

        if fn.__module__ == "__main__":
            wrapper()
        return wrapper

    return wrap if func is None else wrap(func)
