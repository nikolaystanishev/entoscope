import json
from collections.abc import Iterable, Iterator
from itertools import combinations, repeat
from typing import Any, TypeVar

import pandas as pd


def flatten(data: list | tuple | tuple[list | tuple]) -> list | tuple:
    return sum(data, type(data)())


def range_combinations(data: list, min_range: int, max_range: int) -> list:
    return flatten([list(combinations(data, i)) for i in range(min_range, max_range + 1)])


def dataframe_to_list(df: pd.DataFrame) -> list[list]:
    return list(map(lambda el: el[1].to_list(), df.iterrows()))


def zip_fill(*args: Any) -> Iterable:
    max_len = max(map(len, args))
    assert all(map(lambda el: len(el) == max_len or len(el) == 0 or len(el) == 1, args))

    return zip(
        *[arg if len(arg) == max_len else repeat(list(arg)[0] if len(arg) != 0 else None, max_len) for arg in args],
        strict=True,
    )


def to_dict(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.__dict__))


T = TypeVar("T")


def batch_generator(data: list[T], batch_size: int) -> Iterator[list[T]]:
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]
