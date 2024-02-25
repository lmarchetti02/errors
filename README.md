# Uncertainties

A data-analysis-oriented library for propagating errors easily.

## Description

This library allows to propagate errors on lab measures easily. The user just needs to:
1. specify the function relating the physical quantities together: $G \equiv G(x_1,x_2,...,x_n)$;
2. input the arrays with the measures of each quantity, and their errors:
    $$\begin{split}
        &\bar{\textbf{x}}_1=(\bar{x}_{1,1},...,\bar{x}_{1,k}), \ \bar{\textbf{x}}_2,...\\
        &\boldsymbol{\sigma}_1=(\sigma_{1,1},..., \ \sigma_{1,k}), \boldsymbol{\sigma}_2,...
    \end{split}$$
3. get back the array with the propagated errors, where each element is given by the following:
   $$ \sigma_{G} = \sqrt{\sum_{j=1}^n\left( \frac{\partial G}{\partial x_j}(\bar{\textbf{x}}) \right)^2\sigma_j^2} $$

## Installation

Follow the steps below to install _Plotter_ (Unix-like systems):

1. Clone the repository to a directory in your system.
2. Open the terminal and navigate to said folder.
3. Run the command
   ```bash
   python3 -m install
   ```
4. A _dist_ directory will be created, containing a .whl file.
5. To install the package run
   ```bash
   pip3 install /path/to/whl/file
   ```
6. If the installation was successful, _Plotter_ can be
   imported simply by
   ```python
   import uncertainties as u
   ```

__Note__: It is possible that the commands to use are `python` and `pip`, instead of, respectively, `python3` and `pip3`.

## Usage

The library has two modes.
1. For products, quotients and sums of physical quantities, highly optimized algorithms can be used. Therefore, the user should use directly the `products()`, `quotients()` and `sums()` functions.
2. For more complicated functions, a less optimized-but equally fast-algorithm can be used. I these cases, the user should use the `functions()` function.

### Example

The following example shows all the main features of the library.

```python
import numpy as np
import uncertainties as u

size = 1_000_000

# first quantity
x = np.random.binomial(100, 0.3, size)
x_err = np.random.uniform(0.01, 0.03, size)

# second quantity
y = np.random.normal(0, 1, size)
y_err = np.random.uniform(0.1, 0.2, size)

# product
p = x * y
p_err = u.products(x, y, x_err, y_err)

# quotient
q = x / y
q_err = u.quotients(x, y, x_err, y_err)

# sum
s = x + y
s_err = u.sums(x, y, x_err, y_err)

# function
f = x**2 + 2 * y * x
f_err = u.functions(("x", "y"), "x^2 + 2*y*x", [x, y], [x_err, y_err])
```

## License

See the [LICENCE](LICENCE) file for licence rights and limitations.
