import argparse
import dataclasses
import types
import typing
from collections.abc import Sequence


class DictAction(argparse.Action):
    """Collect ``key:value`` tokens into a dict, converting each side with ``key_type``/``value_type``.

    Tokens split on the first ``:`` only, so values may contain colons. Each occurrence of the
    option builds a fresh dict (the last one wins), leaving any default dict untouched.
    """

    def __init__(self, option_strings, dest, key_type=str, value_type=str, **kwargs):
        self._key_type = key_type
        self._value_type = value_type
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            return
        result = {}
        for item in values:
            key, sep, value = item.partition(":")
            if not sep:
                raise argparse.ArgumentError(self, f"expected KEY:VALUE, got {item!r}")
            try:
                result[self._key_type(key)] = self._value_type(value)
            except ValueError as e:
                raise argparse.ArgumentError(self, f"invalid KEY:VALUE {item!r}: {e}") from e
        setattr(namespace, self.dest, result)


class TupleAction(argparse.Action):
    """Collect values into a tuple, converting each position with its own type.

    ``element_types`` mirrors ``typing.get_args`` of a ``tuple`` annotation: ``(int, str)``
    converts exactly two values positionally, ``(int, Ellipsis)`` converts any number of values
    with ``int``.
    """

    def __init__(self, option_strings, dest, element_types=(str, ...), **kwargs):
        self._element_types = tuple(element_types)
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            return
        homogeneous = self._element_types[-1] is Ellipsis
        value_types = (self._element_types[0],) * len(values) if homogeneous else self._element_types
        try:
            result = tuple(t(v) for t, v in zip(value_types, values, strict=True))
        except ValueError as e:
            raise argparse.ArgumentError(self, f"invalid value: {e}") from e
        setattr(namespace, self.dest, result)


def _unwrap_optional(annotation: type) -> type:
    """Reduce ``X | None`` to ``X``; leave any other annotation untouched."""
    if typing.get_origin(annotation) in {typing.Union, types.UnionType}:
        args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def build_parser(config_cls: type, description: str | None = None) -> argparse.ArgumentParser:
    """Build an ``ArgumentParser`` with one ``--field-name`` option per dataclass field.

    Type, default, and required-ness are derived from the field itself; anything argparse-specific
    that cannot be derived (``help``, ``choices``, ...) is read from ``field.metadata`` and
    overrides the derived kwargs. ``list[X]`` fields become ``nargs="*"`` options whose values are
    converted with ``X``; ``dict[K, V]`` fields become ``nargs="*"`` options taking ``key:value``
    tokens (see ``DictAction``); ``tuple[X, Y]`` fields take exactly that many values converted
    per position, and ``tuple[X, ...]`` any number (see ``TupleAction``). Fields with a default
    get ``(default: ...)`` appended to their help text; put ``%(default)s`` in the metadata help
    to control its placement instead.
    """
    parser = argparse.ArgumentParser(description=description)
    hints = typing.get_type_hints(config_cls)
    for field in dataclasses.fields(config_cls):
        if not field.init:
            continue
        annotation = _unwrap_optional(hints[field.name])
        kwargs: dict[str, typing.Any] = {}
        if annotation is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
        elif annotation is list or typing.get_origin(annotation) is list:
            element_types = typing.get_args(annotation)
            kwargs["type"] = element_types[0] if element_types else str
            kwargs["nargs"] = "*"
        elif annotation is dict or typing.get_origin(annotation) is dict:
            key_type, value_type = typing.get_args(annotation) or (str, str)
            kwargs.update(action=DictAction, key_type=key_type, value_type=value_type, nargs="*", metavar="KEY:VALUE")
        elif annotation is tuple or typing.get_origin(annotation) is tuple:
            element_types = typing.get_args(annotation) or (str, ...)
            kwargs.update(action=TupleAction, element_types=element_types)
            if element_types[-1] is Ellipsis:
                kwargs["nargs"] = "*"
            else:
                kwargs["nargs"] = len(element_types)
                kwargs["metavar"] = tuple(t.__name__.upper() for t in element_types)
        else:
            kwargs["type"] = annotation
        if field.default is not dataclasses.MISSING:
            kwargs["default"] = field.default
        elif field.default_factory is not dataclasses.MISSING:
            kwargs["default"] = field.default_factory()
        else:
            kwargs["required"] = True
        kwargs.update(field.metadata)
        action = parser.add_argument(f"--{field.name.replace('_', '-')}", **kwargs)
        # Adjust action.help after the fact: BooleanOptionalAction appends its own default note on
        # some Python versions, and checking the action (not the kwargs) keeps the suffix single.
        if "default" in kwargs and "%(default)" not in (action.help or ""):
            action.help = f"{action.help} (default: %(default)s)" if action.help else "(default: %(default)s)"
    return parser


def parse_config(config_cls: type, argv: Sequence[str] | None = None, description: str | None = None):
    """Parse ``argv`` (``sys.argv[1:]`` when ``None``) into an instance of ``config_cls``."""
    parser = build_parser(config_cls, description)
    return config_cls(**vars(parser.parse_args(argv)))
