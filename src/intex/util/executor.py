from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def none_executor(func: Callable[[T], R], data: T, default: R = None) -> R:
    return func(data) if data is not None else default
