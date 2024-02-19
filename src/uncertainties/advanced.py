import numpy as np
import sympy as sp
from sympy.parsing.mathematica import parse_mathematica


def propagate(var_names: tuple[str], f_expression: str, values: np.ndarray, errors: np.ndarray) -> np.ndarray:
    """ """

    variables = [sp.symbols(f"{var_names[i]}") for i in range(len(var_names))]
    function = parse_mathematica(f_expression)

    derivatives = []
    for var in variables:
        derivatives.append(sp.lambdify(variables, sp.diff(function, var)))
