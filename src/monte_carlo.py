"""
Monte Carlo simulation and European option pricing
under the Bates stochastic volatility jump-diffusion model.

The Bates model combines:
1. Heston stochastic volatility dynamics
2. Merton jump diffusion dynamics

This module provides:
- Bates path simulation using Euler discretization
- European call option pricing via Monte Carlo
- Visualization of simulated asset paths
"""

import numpy as np
import matplotlib.pyplot as plt
def simulate_bates_paths(
    S0: float,
    v0: float,
    r: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    jump_intensity: float,
    jump_mean: float,
    jump_std: float,
    T: float,
    steps: int,
    n_paths: int,
    seed: int = None
):
    """
    Simulate asset price and variance paths under the Bates model.
    Parameters
    ----------
    S0 : float
        Initial asset price.
    v0 : float
        Initial variance.
    r : float
        Risk-free interest rate.
    kappa : float
        Mean reversion speed of variance process.
    theta : float
        Long-run average variance.
    sigma_v : float
        Volatility of volatility.
    rho : float
        Correlation between asset and variance Brownian motions.
    jump_intensity : float
        Average number of jumps per year.
    jump_mean : float
        Mean of jump size in log-space.
    jump_std : float
        Standard deviation of jump size.
    T : float
        Time to maturity in years.
    steps : int
        Number of time discretization steps.
    n_paths : int
        Number of Monte Carlo simulation paths.
    seed : int, optional
        Random seed for reproducibility.
    Returns
    -------
    S : ndarray
        Simulated asset price paths.
    v : ndarray
        Simulated variance paths.
    t : ndarray
        Time grid.
    """
    rng = np.random.default_rng(seed)
    dt = T / steps
    t = np.linspace(0.0, T, steps + 1)
    # Jump compensator term:
    # E[e^J - 1]
    k = np.exp(jump_mean + 0.5 * jump_std**2) - 1
    # Generate correlated Brownian motions
    Z1 = rng.standard_normal((n_paths, steps))
    Z2 = rng.standard_normal((n_paths, steps))
    dW1 = np.sqrt(dt) * Z1
    dW2 = np.sqrt(dt) * (
        rho * Z1 + np.sqrt(1 - rho**2) * Z2
    )
    # Store simulated asset price and variance trajectories
    S = np.zeros((n_paths, steps + 1))
    v = np.zeros((n_paths, steps + 1))
    S[:, 0] = S0
    v[:, 0] = v0
    # Euler discretization scheme
    for i in range(steps):
        # Full truncation Euler scheme to ensure non-negative variance
        v[:, i + 1] = np.maximum(v[:, i]+ kappa * (theta - v[:, i]) * dt + sigma_v * np.sqrt(v[:, i]) * dW2[:, i], 0)
        # Poisson-distributed jump arrivals
        N_jump = rng.poisson(jump_intensity * dt, n_paths)
        # Lognormal jump multiplier from Merton jump diffusion
        J = np.exp(jump_mean + jump_std * rng.standard_normal(n_paths))
        # Risk-neutral drift adjusted for:
        # 1. Jump compensator
        # 2. Ito correction term
        drift = (r- jump_intensity * k- 0.5 * v[:, i]) * dt
        diffusion = np.sqrt(v[:, i] * dt) * Z1[:, i]
        jump_term = J ** N_jump
        # Simulated asset price evolution
        S[:, i + 1] = (S[:, i]* np.exp(drift + diffusion)* jump_term)

    return S, v, t


def european_call_payoff(ST, K):
    return np.maximum(ST - K, 0.0)

def monte_carlo_call_price(ST, K, r, T):
    """
    Compute European call option price using Monte Carlo simulation.
    Parameters
    ----------
    ST : ndarray
        Terminal asset prices.
    K : float
        Strike price.
    r : float
        Risk-free interest rate.
    T : float
        Time to maturity.

    Returns
    -------
    price : float
        Estimated call option price.

    std_error : float
        Standard error of Monte Carlo estimator.
    """

    payoffs = european_call_payoff(ST, K)

    price = np.exp(-r * T) * np.mean(payoffs)

    std_error = (
        np.exp(-r * T)
        * np.std(payoffs)
        / np.sqrt(len(payoffs))
    )

    return price, std_error


def plot_simulated_paths(
    time_grid,
    paths,
    n_paths_to_plot=100
):
    plt.figure(figsize=(10, 6))

    for i in range(min(n_paths_to_plot, len(paths))):
        plt.plot(
            time_grid,
            paths[i],
            lw=0.8,
            alpha=0.7
        )

    # Plot average simulated path
    plt.plot(
        time_grid,
        paths.mean(axis=0),
        color="black",
        lw=2,
        label="Average Path"
    )

    plt.title("Bates Model - Simulated Stock Price Paths")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":

    # Example model parameters
    S0 = 100.0
    v0 = 0.04
    r = 0.03

    kappa = 2.0
    theta = 0.04
    sigma_v = 0.3
    rho = -0.5

    jump_intensity = 0.1
    jump_mean = -0.05
    jump_std = 0.15

    T = 1.0
    steps = 1000
    n_paths = 10000

    K = 100

    # Simulate Bates model paths
    S, v, t = simulate_bates_paths(
        S0=S0,
        v0=v0,
        r=r,
        kappa=kappa,
        theta=theta,
        sigma_v=sigma_v,
        rho=rho,
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_std=jump_std,
        T=T,
        steps=steps,
        n_paths=n_paths,
        seed=42
    )

    # Compute Monte Carlo option price
    price, std_error = monte_carlo_call_price(
        ST=S[:, -1],
        K=K,
        r=r,
        T=T
    )

    print(f"Monte Carlo European Call Price: {price:.4f}")
    print(f"Standard Error: {std_error:.6f}")

    # Visualize simulated paths
    plot_simulated_paths(t, S)
