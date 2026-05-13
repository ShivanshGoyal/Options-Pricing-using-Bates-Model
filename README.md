# Bates Model Option Pricing and Calibration

## Overview

This project implements the Bates stochastic volatility jump-diffusion model for European option pricing and market calibration using real NSE option-chain data.

The Bates model extends the Heston stochastic volatility framework by incorporating jump diffusion dynamics, allowing the model to capture both:
- stochastic volatility clustering, and
- sudden discontinuous market movements.

The project combines:
- Monte Carlo simulation using Euler discretization,
- semi-analytical pricing via characteristic functions,
- Fourier inversion methods,
- parameter calibration using market option prices,
- and implied volatility smile analysis.

The implementation is designed as a modular computational finance framework with reusable pricing, calibration, and volatility-analysis components.

---

## Mathematical Framework

Under the risk-neutral measure, the Bates model assumes the asset price dynamics:

$$
dS_t = (r - \lambda k) S_t dt + \sqrt{v_t} S_t dW_t^S + (J - 1) S_t dN_t
$$

where:
- $S_t$ is the underlying asset price,
- $v_t$ is the stochastic variance process,
- $\lambda$ is the jump intensity,
- $J$ is the jump multiplier,
- and $N_t$ is a Poisson jump process.

The stochastic variance evolves according to:

$$
dv_t = \kappa(\theta - v_t)dt + \sigma_v \sqrt{v_t} dW_t^v
$$

where:
- $\kappa$ controls mean reversion speed,
- $\theta$ is the long-run variance,
- $\sigma_v$ is the volatility of volatility,
- and $\rho$ defines the correlation between price and variance shocks.
---

## Features

- Monte Carlo simulation under the Bates model
- Euler discretization for stochastic volatility dynamics
- Semi-analytical pricing using characteristic functions
- Fourier inversion option pricing framework
- NSE option-chain API integration
- Bates model parameter calibration using market data
- Volatility smile generation and comparison
- Market vs model pricing diagnostics
- Error metrics including:
  - MAE
  - RMSE
  - MAPE

---

## Project Structure

```text
bates-model-option-pricing/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── bates_model_demo.ipynb
│
├── src/
│   ├── __init__.py
│   ├── monte_carlo.py
│   ├── characteristic_function.py
│   ├── calibration.py
│   ├── data_fetch.py
│   └── implied_volatility.py
│
├── figures/
│
└── results/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/bates-model-option-pricing.git

cd bates-model-option-pricing
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Fetch NSE Option-Chain Data

```python
from src.data_fetch import (
    fetch_nse_option_chain,
    process_option_chain_data
)

raw_data = fetch_nse_option_chain("NIFTY")

df, spot_price, expiry = process_option_chain_data(raw_data)
```

---

### Calibrate Bates Model

```python
from src.calibration import (
    clean_option_dataset,
    calibrate_bates_model
)

sample_data = clean_option_dataset(
    df,
    spot_price
)

result, calibrated_params = calibrate_bates_model(
    sample_data,
    spot_price
)
```

---

### Generate Model Prices

```python
from src.calibration import generate_model_prices

results_df = generate_model_prices(
    df,
    spot_price,
    calibrated_params
)
```

---

### Compute Implied Volatility Smile

```python
from src.implied_volatility import (
    generate_iv_surface,
    plot_volatility_smile
)

iv_data = generate_iv_surface(
    results_df,
    spot_price
)

plot_volatility_smile(iv_data)
```

---

## Calibration Workflow

```text
NSE Option Chain Data
        ↓
Data Cleaning & Preprocessing
        ↓
Characteristic Function Pricing
        ↓
L-BFGS-B Optimization
        ↓
Calibrated Bates Parameters
        ↓
Model vs Market Comparison
        ↓
Implied Volatility Analysis
```

---

## Results

The project produces:
- simulated stochastic asset paths,
- calibrated Bates model parameters,
- model vs market pricing comparisons,
- and implied volatility smile analysis.

Example outputs include:
- Monte Carlo simulation paths
- Market vs Bates pricing curves
- Volatility smile comparison plots

Example figures can be stored in the `figures/` directory.

---

## Numerical Methods

### Monte Carlo Simulation
The Bates stochastic differential equations are simulated using Euler discretization with:
- correlated Brownian motions,
- stochastic variance evolution,
- and Poisson jump dynamics.

### Characteristic Function Pricing
Semi-analytical pricing is implemented using:
- Fourier inversion,
- risk-neutral characteristic functions,
- and numerical integration.

### Calibration
Model calibration is performed using constrained optimization (`L-BFGS-B`) to minimize pricing error between:
- observed market option prices, and
- Bates model-generated prices.

---

## Future Improvements

Potential extensions include:
- FFT-based option pricing
- Greeks computation
- American option pricing
- GPU acceleration
- Variance reduction techniques
- Local volatility comparison
- SABR and rough volatility extensions

---

## References

1. Bates, D. S. (1996). *Jumps and Stochastic Volatility: Exchange Rate Processes Implicit in Deutsche Mark Options.*

2. Heston, S. L. (1993). *A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options.*

3. Merton, R. C. (1976). *Option Pricing When Underlying Stock Returns Are Discontinuous.*

---

## Disclaimer

This project is intended for educational and research purposes only.  
It should not be interpreted as financial advice or used directly for live trading decisions.
