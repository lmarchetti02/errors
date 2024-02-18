import numpy as np
from numba import njit, objmode


@njit(fastmath=True)
def _product_helper(vars: np.ndarray, errs: np.ndarray) -> np.ndarray:
    """
    This helper functions carries out the heavy computation
    for the `product()` function.

    Parameters
    ---
    vars: numpy.ndarray
        The array containing the arrays with the values of
        the quantities in the product.
    errs: numpy.ndarray
        The array containing the arrays with the values of
        the errors on the quantities in the product.

    Returns
    ---
    The array containing the propagated uncertainties on
    the product of the datasets.
    """
    res = np.empty(len(vars[0]))

    for _ in range(len(vars)):
        var = np.array([vars[i][_] for i in range(len(vars))])
        err = np.array([errs[i][_] for i in range(len(errs))])

        res[_] = var.prod() * np.sqrt(((err / var) ** 2).sum())

    return res


def product(a: np.ndarray, b: np.ndarray, a_err: np.ndarray, b_err: np.ndarray, *args) -> np.ndarray:
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

    variables = np.array([a, b])
    errors = np.array([a_err, b_err])
    if args:
        for i in range(0, len(args), 2):
            variables = np.append(variables, [args[i]], axis=0)
            errors = np.append(errors, [args[i + 1]], axis=0)

    return _product_helper(variables, errors)


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
