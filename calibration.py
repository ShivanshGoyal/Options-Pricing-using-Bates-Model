"""
Bates model calibration utilities.

This module provides:
- Market option dataset preprocessing
- Bates model parameter calibration
- Optimization objective functions
- Error metric computation
- Market vs model price comparison plots

Calibration is performed using NSE option-chain data.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
from src.characteristic_function import (
    bates_option_price
)
def clean_option_dataset(
    df,
    spot_price,
    atm_lower=0.9,
    atm_upper=1.1,
    sample_size=40,
    random_state=42
):
    """
    Clean and preprocess option-chain dataset.

    Parameters
    ----------
    spot_price : float
        Current underlying spot price.
    atm_lower : float
        Lower ATM strike filter bound.
    atm_upper : float
        Upper ATM strike filter bound.
    sample_size : int
        Number of options used for calibration.
    random_state : int
        Random seed for reproducibility.
    Returns
    -------
    pandas.DataFrame
        Cleaned calibration dataset.
    """
    data = df.copy()
    # Remove rows with missing strike prices
    data = data[data["strike"].notnull()].copy()
    # Required columns
    required_columns = [
        "expiry_date",
        "strike",
        "call_price",
        "put_price",
        "maturity",
        "risk_free_rate"
    ]
    data = data[
        [col for col in required_columns if col in data.columns]
    ]
    # Convert columns to numeric format
    numeric_columns = [
        "strike",
        "call_price",
        "maturity",
        "risk_free_rate"
    ]
    for col in numeric_columns:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )
    # Focus calibration around ATM region
    atm_data = data[
        (data["strike"] > atm_lower * spot_price)
        & (data["strike"] < atm_upper * spot_price)
    ]
    # Random subset for computational efficiency
    sample_data = atm_data.sample(
        min(sample_size, len(atm_data)),
        random_state=random_state
    )
    return sample_data
def calibration_objective(
    params,
    data,
    spot_price
):
    """
    Objective function for Bates model calibration.
    Minimizes mean squared pricing error between
    market prices and model prices.
    Parameters
    ----------
    params : list
        Bates model parameter vector.
    data : pandas.DataFrame
        Calibration dataset.
    spot_price : float
        Current underlying spot price.
    Returns
    -------
    float
        Mean squared pricing error.
    """
    (kappa,theta,sigma_v,rho,v0,jump_intensity,jump_mean,jump_std) = params
    errors = []
    for _, row in data.iterrows():
        K = row["strike"]
        T = row["maturity"]
        r = row["risk_free_rate"]
        market_price = row["call_price"]
        try:
            model_price, _ = bates_option_price(
                S0=spot_price,
                K=K,
                r=r,
                T=T,
                kappa=kappa,
                theta=theta,
                sigma_v=sigma_v,
                rho=rho,
                v0=v0,
                jump_intensity=jump_intensity,
                jump_mean=jump_mean,
                jump_std=jump_std
            )
            errors.append(
                (model_price - market_price) ** 2
            )
        except Exception:
            continue
    # Prevent empty-error optimization failure
    if len(errors) == 0:
        return 1e10
    return np.mean(errors)
def calibrate_bates_model(
    data,
    spot_price,
    initial_guess=None,
    bounds=None
):
    if initial_guess is None:
        initial_guess = [
            2.0,    # kappa
            0.04,   # theta
            0.3,    # sigma_v
            -0.5,   # rho
            0.04,   # v0
            0.1,    # lambda
            -0.1,   # muJ
            0.2     # sigmaJ
        ]
    if bounds is None:
        bounds = [
            (0.1, 10),      # kappa
            (0.001, 1),     # theta
            (0.05, 1),      # sigma_v
            (-0.99, 0.99),  # rho
            (0.001, 1),     # v0
            (0.001, 1),     # lambda
            (-0.5, 0.5),    # muJ
            (0.01, 1)       # sigmaJ
        ]
    result = minimize(
        calibration_objective,
        initial_guess,
        args=(data, spot_price),
        bounds=bounds,
        method="L-BFGS-B"
    )
    parameter_names = [
        "kappa",
        "theta",
        "sigma_v",
        "rho",
        "v0",
        "jump_intensity",
        "jump_mean",
        "jump_std"
    ]
    calibrated_params = dict(
        zip(parameter_names, result.x)
    )
    return result, calibrated_params


def generate_model_prices(
    data,
    spot_price,
    calibrated_params
):
    """
    Generate Bates model prices using calibrated parameters.

    Parameters
    ----------
    data : pandas.DataFrame
        Option-chain dataset.
    spot_price : float
        Current underlying spot price.
    calibrated_params : dict
        Calibrated Bates model parameters.
    Returns
    -------
    pandas.DataFrame
        Dataset with model-generated prices.
    """
    output = data.copy()
    output["model_price"] = output.apply(
        lambda row: bates_option_price(
            S0=spot_price,
            K=row["strike"],
            r=row["risk_free_rate"],
            T=row["maturity"],
            **calibrated_params
        )[0],
        axis=1
    )

    return output
def compute_error_metrics(
    market_prices,
    model_prices
):
    """
    Compute model calibration error metrics.
    Parameters
    ----------
    market_prices : ndarray
        Observed market option prices.
    model_prices : ndarray
        Bates model-generated prices.
    Returns
    -------
    dict
        Dictionary containing MAE, RMSE, and MAPE.
    """
    mae = mean_absolute_error( market_prices, model_prices)
    rmse = np.sqrt(mean_squared_error(market_prices,model_prices))
    epsilon = 1e-10
    mape = np.mean(np.abs((market_prices - model_prices)/ (market_prices + epsilon))) * 100
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }
def plot_model_vs_market(
    data
):
    """
    Plot Bates model prices against market option prices.
    Parameters
    ----------
    data : pandas.DataFrame
        Dataset containing market and model prices.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(
        data["strike"],
        data["call_price"],
        label="Market Call Price",
        color="blue",
        s=60,
        alpha=0.6
    )

    plt.plot(
        data["strike"],
        data["model_price"],
        label="Bates Model Price",
        color="red",
        linewidth=2.5
    )
    plt.title(
        "Bates Model vs Market Option Prices"
    )
    plt.xlabel("Strike Price")
    plt.ylabel("Call Option Price")
    plt.grid(
        True,
        linestyle="--",
        alpha=0.6
    )
    plt.legend()
    plt.show()

if __name__ == "__main__":

    print(
        "Calibration module loaded successfully."
    )
    print(
        "Use this module together with "
        "data_fetch.py and "
        "characteristic_function.py"
    )
