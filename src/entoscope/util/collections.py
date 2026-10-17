from collections.abc import Iterator, Sequence
from itertools import combinations
from typing import TypeVar

import pandas as pd


def flatten(data: list | tuple | tuple[list | tuple]) -> list | tuple:
    return sum(data, type(data)())


def range_combinations(data: list, min_range: int, max_range: int) -> list:
    return flatten([list(combinations(data, i)) for i in range(min_range, max_range + 1)])


def dataframe_to_list(df: pd.DataFrame) -> list[list]:
    return list(map(lambda el: el[1].to_list(), df.iterrows()))


T = TypeVar("T")


def broadcast_length(*lengths: int) -> int:
    """The length sequences of ``lengths`` pair up to, one of a single element repeated along the others.

    Raises ``ValueError`` when two of them are longer than one and differ.
    """
    longer = set(lengths) - {1}
    if len(longer) > 1:
        raise ValueError(f"Cannot pair sequences of {' and '.join(map(str, sorted(longer)))} elements")
    return longer.pop() if longer else 1


def zip_broadcast(*sequences: Sequence[T]) -> list[tuple[T, ...]]:
    """Zip ``sequences`` element by element, one of a single element repeated along the others."""
    length = broadcast_length(*map(len, sequences))
    return list(
        zip(*(list(sequence) * length if len(sequence) == 1 else sequence for sequence in sequences), strict=True)
    )


def batch_generator(data: list[T], batch_size: int) -> Iterator[list[T]]:
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]
