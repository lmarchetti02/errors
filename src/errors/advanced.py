import numpy as np
import sympy as sp
from sympy.parsing.mathematica import parse_mathematica


def functions(
    var_names: tuple[str, ...],
    f_expression: str,
    values: list[np.ndarray],
    errors: list[np.ndarray],
) -> np.ndarray:
    """
    This function propagates the uncertainties in the case of functions
    of physical quantities, that is, `G = G(A, B, ...)`.

    Parameters
    ---
    var_names: tuple[str]
        The tuple containing the names of the function
        variables as strings.
    f_expression: str
        The string with the expression of the function
        in the Mathematica syntax.
    values: list
        The list containing the arrays of values of the
        function variables.
    errors: list
        The list containing the arrays of values of the
        errors on the function variables.

    IMPORTANT: This function considers the function variables as
               independent, that is, the covariances between said
               variables are considered 0.
    """

    variables = [sp.symbols(f"{var_names[i]}") for i in range(len(var_names))]
    function = parse_mathematica(f_expression)

    derivatives = []
    for var in variables:
        derivatives.append(sp.lambdify(variables, sp.diff(function, var)))

    derivatives = np.array(derivatives)

    n_elements = len(values[0])
    values = np.array(values).transpose()
    errors = np.array(errors).transpose()

    res = np.empty(n_elements)

    for i in range(n_elements):
        der_values = np.array([der(*values[i]) for der in derivatives])
        res[i] = np.sqrt(((der_values * errors[i]) ** 2).sum())

    return res


def functions_single(
    var_names: tuple[str],
    f_expression: str,
    values: list[float],
    errors: list[float],
) -> np.ndarray:
    """
    This function propagates the uncertainties in the case of functions
    of physical quantities, that is, `G = G(A, B, ...)`.

    It is analogous to `functions()`, but it takes single measures
    instead of arrays of them.

    Parameters
    ---
    var_names: tuple[str]
        The tuple containing the names of the function
        variables as strings.
    f_expression: str
        The string with the expression of the function
        in the Mathematica syntax.
    values: list
        The list containing the values of the function variables.
    errors: list
        The list containing the values of the errors on the
        function variables.

    IMPORTANT: This function considers the function variables as
               independent, that is, the covariances between said
               variables are considered 0.
    """

    variables = [sp.symbols(f"{var_names[i]}") for i in range(len(var_names))]
    function = parse_mathematica(f_expression)

    derivatives = []
    for var in variables:
        derivatives.append(sp.lambdify(variables, sp.diff(function, var)))

    derivatives = np.array(derivatives)

    values = np.array(values)
    errors = np.array(errors)

    der_values = np.array([der(*values) for der in derivatives])
    return np.sqrt(((der_values * errors) ** 2).sum())
