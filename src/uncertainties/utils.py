from numba import njit


@njit
def check_sizes(a_list: list) -> None:
    """
    This function checks whether an array is composed
    of equal-sized arrays.

    Parameters
    ---
    a_list: numpy.ndarray
        The array of arrays.
    """

    for i, arr in enumerate(a_list):
        for j in range(i, len(a_list)):
            if len(arr) != len(a_list[j]):
                raise ValueError("The arrays don't have the same sizes")
