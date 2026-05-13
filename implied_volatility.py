"""
Implied volatility utilities and volatility smile analysis.

This module provides:
- Black-Scholes call option pricing
- Implied volatility extraction
- Volatility smile generation
- Market vs model IV comparison

Used for evaluating Bates model calibration quality.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

def black_scholes_call_price(S,K,r,T,sigma):
    if sigma <= 0 or T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K)+ (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = (S * norm.cdf(d1)- K * np.exp(-r * T) * norm.cdf(d2))
    return price

def implied_volatility_error(sigma,S,K,r,T,market_price):
    """
    Objective function for implied volatility inversion.
    Parameters
    ----------
    sigma : float
        Volatility guess.
    market_price : float
        Observed option price.
    Returns
    -------
    float
        Pricing error.
    """
    model_price = black_scholes_call_price(S, K,r,T,sigma)
    return model_price - market_price
  
def compute_implied_volatility(S,K,r,T,market_price,lower_bound=1e-4,upper_bound=3.0):
    """
    Compute implied volatility using Brent root-finding.
    Parameters
    ----------
    S : float
        Current asset price.
    K : float
        Strike price.
    r : float
        Risk-free interest rate.
    T : float
        Time to maturity.
    market_price : float
        Observed option market price.
    lower_bound : float
        Lower volatility search bound.
    upper_bound : float
        Upper volatility search bound.
    Returns
    -------
    float
        Implied volatility estimate.
    """
    try:
        implied_vol = brentq(implied_volatility_error,lower_bound,upper_bound,args=(S,K,r,T,market_price))
        return implied_vol
    except Exception:
        return np.nan

def generate_iv_surface(data,spot_price):
    """
    Compute market and model implied volatilities.
    Parameters
    ----------
    data : pandas.DataFrame
        Dataset containing:
        - strike
        - call_price
        - model_price
        - maturity
        - risk_free_rate
    spot_price : float
        Current underlying spot price.
    Returns
    -------
    pandas.DataFrame
        Dataset with implied volatility columns added.
    """
    output = data.copy()
    market_ivs = []
    model_ivs = []
    for _, row in output.iterrows():
        K = row["strike"]
        T = row["maturity"]
        r = row["risk_free_rate"]
        # Market implied volatility
        market_iv = compute_implied_volatility( S=spot_price, K=K, r=r, T=T, market_price=row["call_price"] )
        # Bates model implied volatility
        model_iv = compute_implied_volatility( S=spot_price, K=K, r=r, T=T, market_price=row["model_price"] )
        market_ivs.append(market_iv)
        model_ivs.append(model_iv)
    output["market_iv"] = market_ivs
    output["model_iv"] = model_ivs
    return output


def plot_volatility_smile(data):
    """
    Plot market vs model implied volatility smile.
    Parameters
    ----------
    data : pandas.DataFrame
        Dataset containing:
        - strike
        - market_iv
        - model_iv
    """
    plot_data = data.dropna(
        subset=["market_iv", "model_iv"]
    )
    plt.figure(figsize=(10, 6))
    # Market implied volatility
    plt.scatter(plot_data["strike"],plot_data["market_iv"],label="Market IV",color="blue"
    )
    # Bates model implied volatility
    plt.plot( plot_data["strike"], plot_data["model_iv"], label="Model IV", color="red", linewidth=2 )
    plt.title("Volatility Smile: Market vs Bates Model")
    plt.xlabel("Strike Price")
    plt.ylabel("Implied Volatility")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":

    print(
        "Implied volatility module loaded successfully."
    )

    print(
        "Use this module together with "
        "calibration outputs to analyze "
        "volatility smiles."
    )
