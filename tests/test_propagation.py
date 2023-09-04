from errors import propagation as p
from sympy.parsing.mathematica import parse_mathematica
from numpy import array, array_equal

p.Functions.activate_logging(status=False)


def test_derivate_error():
    variabili = p.Functions.def_variabili(("x", "y"))
    G = parse_mathematica("x^2 + y^2")
    derivate = p.Functions.derivate(G, variabili, array([0.1, 0.2, 0.3]))

    assert array_equal(derivate, [])


def test_derivate():
    variabili = p.Functions.def_variabili(("x", "y"))
    G = parse_mathematica("x^2 + y^2")
    derivate = p.Functions.derivate(G, variabili, array([0.1, 0.2]))

    assert array_equal(derivate, array([0.2, 0.4]))
