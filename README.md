# Financial-Pricing-Models

[![Repo Size](https://img.shields.io/badge/size-small-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)]()
[![Language: Python](https://img.shields.io/badge/language-Python-yellow.svg)]()

A collection of financial option pricing models and utilities implemented in Python, primarily focused on the Black–Scholes family of models, binomial trees, Monte Carlo simulation, and related analytics (Greeks, implied volatility, calibration helpers). This repository is intended as a learning resource and a lightweight library for pricing and risk analysis of derivative instruments.

Table of Contents

- Project Overview
- Features
- Models Implemented
- Quickstart
- Installation
- Usage Examples
  - Black–Scholes analytic price
  - Binomial tree
  - Monte Carlo with antithetic variates
  - Greeks and implied volatility
- API / Project Structure
- Development & Testing
- Notebooks & Examples
- Contributing
- Roadmap
- References
- License
- Acknowledgements

## Project Overview

This project gathers common pricing methodologies used in academic and industry contexts for pricing European and American options as well as more advanced simulation-based approaches.

Goals:

- Provide clear, readable implementations of classic pricing models.
- Offer examples and notebooks for learning and experimentation.
- Provide utilities for calibration, risk measures (Greeks), and plotting.

## Features

- Black–Scholes closed-form pricing for European options
- Greeks computation (Delta, Gamma, Vega, Theta, Rho)
- Implied volatility via root-finding
- Cox-Ross-Rubinstein / Recombinant binomial tree for European & American options
- Monte Carlo simulation with variance reduction (antithetic variates, control variates)
- Simple calibration helpers and parameter estimation
- Example scripts and Jupyter notebooks for visualization and experimentation

## Models Implemented

- Black–Scholes-Merton analytic formulas for European call and put
- Binomial Tree (CRR) — European & American options
- Monte Carlo — plain, antithetic, and control variate
- Implied volatility solver (Brent or Newton method)
- Basic dividend and yield support (continuous yield approximation)
- Numeric Greeks (finite difference) and analytic Greeks where available

## Quickstart

1. Clone the repository:

   ```bash
   git clone https://github.com/qquella/Financial-Pricing-Models.git
   cd Financial-Pricing-Models
   ```

2. Create a Python virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   .venv\Scripts\activate      # Windows
   pip install -r requirements.txt
   ```

3. Run an example script:
   ```bash
   python examples/black_scholes_example.py
   ```

## Installation

Recommended Python versions: 3.8+

Install from PyPI (if published) or from source:

From source:

```bash
pip install -e .
```

Or install only the runtime dependencies:

```bash
pip install numpy scipy pandas matplotlib jupyter
```

(If you use numba for acceleration in heavy Monte Carlo tasks, add `numba`.)

## Usage Examples

1. Black–Scholes analytic price (Python API example)

```python
from pricing.black_scholes import bs_price, bs_greeks

S = 100.0       # spot price
K = 100.0       # strike
r = 0.01        # annual risk-free rate
sigma = 0.2     # volatility
T = 0.5         # time to maturity in years

call_price = bs_price(option_type='call', S=S, K=K, r=r, sigma=sigma, T=T)
greeks = bs_greeks(option_type='call', S=S, K=K, r=r, sigma=sigma, T=T)

print(f"Call price: {call_price:.4f}")
print("Greeks:", greeks)
```

2. Binomial Tree (CRR) example:

```python
from pricing.binomial import cr_tree_price

price = cr_tree_price(option_type='put', S=100, K=105, r=0.01, sigma=0.25, T=1, N=200, american=True)
print(f"American put price (CRR, N=200): {price:.4f}")
```

3. Monte Carlo with antithetic variates:

```python
from pricing.monte_carlo import mc_price

price, stderr = mc_price(option_type='call', S=100, K=100, r=0.01, sigma=0.2, T=1, n_paths=100_000, antithetic=True)
print(f"MC price: {price:.4f} ± {1.96*stderr:.4f} (95% CI)")
```

4. Implied volatility solver:

```python
from pricing.implied_vol import implied_vol

market_price = 2.5
iv = implied_vol(option_type='call', S=100, K=100, r=0.01, T=0.5, price=market_price, initial_guess=0.2)
print(f"Implied vol: {iv:.4%}")
```

## API / Project Structure

A suggested logical layout (actual repo files may vary):

```
pricing/
  __init__.py
  black_scholes.py      # analytic BS price and Greeks
  binomial.py           # CRR binomial tree implementation
  monte_carlo.py        # Monte Carlo drivers and variance reduction
  implied_vol.py        # implied volatility root-finding
  utils.py              # helpers (day count, input validation)
examples/
  black_scholes_example.py
  binomial_example.py
  monte_carlo_example.py
notebooks/
  bs_visualization.ipynb
  mc_convergence.ipynb
tests/
  test_black_scholes.py
  test_binomial.py
requirements.txt
README.md
```

If you add modules, please keep naming consistent and well documented.

## Development & Testing

- Run unit tests with pytest:

  ```bash
  pytest -q
  ```

- Run linters (optional):

  ```bash
  pip install flake8 black
  black .
  flake8 .
  ```

- Add new tests for any new pricing engine or utility function.

## Notebooks & Examples

The `notebooks/` directory contains interactive examples showing:

- Visualizing the Black–Scholes call/put price surfaces across S and sigma
- Monte Carlo convergence and variance reduction effects
- American option early exercise boundaries from binomial trees

## Contributing

Contributions are welcome — please follow these guidelines:

1. Open an issue describing the feature or bug before opening a PR (so we can discuss scope).
2. Create a branch for your work: `git checkout -b feat/your-feature`.
3. Add tests for new features and make sure existing tests pass.
4. Follow the repository's style (PEP8) and use clear docstrings for new functions.
5. Open a pull request with a descriptive title and summary of changes.

Coding style

- PEP 8 / PEP 257 docstrings.
- Type hints where appropriate.

## Roadmap

Planned enhancements:

- Add local volatility models and Heston model analytic/FFT approximations.
- Faster Monte Carlo using GPU acceleration (CuPy) option.
- Calibration utilities to market implied vol surfaces.
- Packaging and publishing to PyPI.

## References

- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities.
- Hull, J. C. (Options, Futures, and Other Derivatives).
- Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). Option pricing: A simplified approach.
- Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering.

## License

This project is licensed under the MIT License — see the LICENSE file for details.

## Acknowledgements

- Implementations inspired by standard textbooks and open-source educational projects.
- Thanks to contributors and those reporting issues.

## Contact / Maintainer

Maintained by: qquella (https://github.com/qquella)

If you'd like changes to wording, structure, or additional examples (like volatility surfaces, Greeks tables, or performance tips), tell me what to add and I will update the README accordingly.
