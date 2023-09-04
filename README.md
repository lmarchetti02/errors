# Errors

A library for propagating uncertainties easily.

## Description

Given a set of direct measurements _(x_1,x_2,...,x_n)_ with their errors _(e_1,e_2,...,e_n)_, and a
function _G=G(x_1,x_2,...,x_n)_; the library calculates the error on the value of _G(x_1,x_2,...,x_n)_
due to the uncertainties _(e_1,e_2,...,e_n)_.

## Installation

Follow the steps below to install _grada_:

1. Clone the repository in a directory:

   ```bash
    git clone https://github.com/lmarchetti02/grada
   ```

2. Install `wheel`, `setuptools`, `twine`

   ```bash
    pip3 install wheel
   ```

   ```bash
    pip3 install setuptools
   ```

   ```bash
    pip3 install twine
   ```

3. Build the library by running:

   ```bash
    python3 setup.py bdist_wheel
   ```

   This will create a folder named 'dist' in the working directory, which contains a file
   with '.whl' extension

4. Install the library by running:

   ```bash
   pip install /path/to/wheelfile.whl
   ```

## Example

For a basic graph, try the following piece of code.

```python
from errors import propagation as p
from numpy import array

direct_measurements = array([0.1, 0.2, 0.3])
errors = array([0.01, 0.05, 0.1])
G = "x^2 + y^2 + z^2"

G_error = propagazione_errori(
        ("x", "y", "z"),
        G,
        direct_measurements,
        errors,
        True,
    )
```

## License

See the [LICENCE](LICENCE) file for licence rights and limitations.
