import numpy as np
from numba import njit, objmode


@njit(fastmath=True)
def product(a: np.ndarray, b: np.ndarray, a_err: np.ndarray, b_err: np.ndarray) -> np.ndarray:
    """
    This function propagates the uncertainties in the case of products of
    physical quantities, that is, `G = AB`.

    Parameters
    ---
    a: numpy.ndarray
        The array with the values of the first quantity.
    b: numpy.ndarray
        The array with the values of the second quantity.
    a_err: numpy.ndarray
        The array with the errors on the values of the first quantity.
    b_err: numpy.ndarray
        The array with the errors on the values of the second quantity.
    """

    errors = np.empty(len(a))

    for _ in range(len(a)):
        errors[_] = (a[_] * b[_]) * np.sqrt((a_err[_] / a[_]) ** 2 + (b_err[_] / b[_]) ** 2)

    return errors


@njit(fastmath=True)
def quotient(a: np.ndarray, b: np.ndarray, a_err: np.ndarray, b_err: np.ndarray) -> np.ndarray:
    """
    This function propagates the uncertainties in the case of quotients of
    physical quantities, that is, `G = A/B`.

    Parameters
    ---
    a: numpy.ndarray
        The array with the values of the first quantity.
    b: numpy.ndarray
        The array with the values of the second quantity.
    a_err: numpy.ndarray
        The array with the errors on the values of the first quantity.
    b_err: numpy.ndarray
        The array with the errors on the values of the second quantity.
    """

    errors = np.empty(len(a))

    for _ in range(len(a)):
        errors[_] = (a[_] / b[_]) * np.sqrt((a_err[_] / a[_]) ** 2 + (b_err[_] / b[_]) ** 2)

    return errors
