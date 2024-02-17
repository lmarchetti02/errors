from numba import njit
from typing import NoReturn


@njit
def check_sizes(a_list: list) -> NoReturn:
    for i, arr in enumerate(a_list):
        for j in range(i, len(a_list)):
            if len(arr) != len(a_list[j]):
                raise ValueError("The arrays don't have the same sizes")
