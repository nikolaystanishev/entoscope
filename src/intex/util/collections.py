from itertools import combinations


def flatten(data: list | tuple | tuple[list | tuple]) -> list | tuple:
    return sum(data, type(data)())


def range_combinations(data: list, min_range: int, max_range: int) -> list:
    return flatten([list(combinations(data, i)) for i in range(min_range, max_range + 1)])
