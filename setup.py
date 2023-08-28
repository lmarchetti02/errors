from setuptools import find_packages, setup

setup(
    name="errors",
    packages=find_packages(include=["errors"]),
    version="0.1.0",
    description="A library for propagating uncertainties easily.",
    author="Luca Marchetti",
    license="MIT",
    install_requires=["sympy", "logging"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest>=4.4.1"],
    test_suite="tests",
)
