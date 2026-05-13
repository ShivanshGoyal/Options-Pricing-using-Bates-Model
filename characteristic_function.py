"""
Semi-analytical option pricing under the Bates model
using characteristic function methods.

The Bates model combines:
1. Heston stochastic volatility
2. Merton jump diffusion

This module implements:
- Bates characteristic function
- Risk-neutral probability integrals
- European call and put option pricing

Pricing is performed using Fourier inversion techniques.
"""
import numpy as np
from scipy.integrate import quad
def bates_characteristic_function(
    u,
    S0,
    r,
    T,
    kappa,
    theta,
    sigma_v,
    rho,
    v0,
    jump_intensity,
    jump_mean,
    jump_std
):
    """
    Compute the Bates model characteristic function.
    Parameters
    ----------
    u : complex
        Complex integration variable.
    S0 : float
        Initial asset price.
    r : float
        Risk-free interest rate.
    T : float
        Time to maturity.
    kappa : float
        Mean reversion speed of variance process.

    theta : float
        Long-run average variance.
    sigma_v : float
        Volatility of volatility.
    rho : float
        Correlation between asset and variance processes.
    v0 : float
        Initial variance.
    jump_intensity : float
        Average jump arrivals per year.
    jump_mean : float
        Mean jump size in log-space.
    jump_std : float
        Standard deviation of jump size.
    Returns
    -------
    complex
        Value of the Bates characteristic function.
    """

    i = 1j
    # Heston stochastic volatility component
    xi = kappa - rho * sigma_v * i * u
    d = np.sqrt(
        (rho * sigma_v * i * u - xi) ** 2 - sigma_v**2 * (-i * u - u**2)
    )
    g = (
        (xi - rho * sigma_v * i * u - d)/ (xi - rho * sigma_v * i * u + d)
    )
    C = (r * i * u * T+ (kappa * theta / sigma_v**2)* 
         ((xi - rho * sigma_v * i * u - d) * T- 2 * np.log((1 - g * np.exp(-d * T))/ (1 - g))))
    D = (
        (xi - rho * sigma_v * i * u - d)/ sigma_v**2) * 
        ((1 - np.exp(-d * T))/ (1 - g * np.exp(-d * T)))
    heston_component = np.exp(
        C + D * v0 + i * u * np.log(S0)
    )
    # Jump diffusion component
    jump_component = np.exp(jump_intensity* T* (np.exp(i * u * jump_mean- 0.5 * jump_std**2 * u**2) - 1))
    return heston_component * jump_component


def calculate_probabilities(S0,K,r,T,kappa,theta,sigma_v,rho,v0,jump_intensity,jump_mean,jump_std):
    """
    Compute risk-neutral probabilities P1 and P2
    using Fourier inversion.
    P1 : float
        Probability term associated with stock component.
    P2 : float
        Probability term associated with strike component.
    """
    def integrand_P2(u):

        phi_u = bates_characteristic_function(u,S0,r,T,kappa,theta,sigma_v,rho,v0,jump_intensity,jump_mean,jump_std)
        value = (np.exp(-1j * u * np.log(K))* phi_u/ (1j * u))
        return np.real(value)

    def integrand_P1(u):

        phi_u_minus_i = bates_characteristic_function(u - 1j,S0,r,T,kappa,theta,sigma_v,rho,v0,jump_intensity,jump_mean,jump_std)
        
      phi_minus_i = bates_characteristic_function(-1j,S0, r, T, kappa, theta, sigma_v, rho, v0, jump_intensity, jump_mean, jump_std)

        value = (np.exp(-1j * u * np.log(K))* phi_u_minus_i/ (1j * u * phi_minus_i))
        return np.real(value)

    # Numerical integration over Fourier domain
    integral_P2, _ = quad(integrand_P2,0,np.inf,limit=200)

    integral_P1, _ = quad(integrand_P1, 0, np.inf, limit=200)

    P2 = 0.5 + (1 / np.pi) * integral_P2
    P1 = 0.5 + (1 / np.pi) * integral_P1

    return P1, P2


def bates_option_price(S0,K, r, T, kappa, theta, sigma_v, rho, v0, jump_intensity, jump_mean, jump_std):
   
    P1, P2 = calculate_probabilities(S0,K,r,T,kappa,theta,sigma_v,rho,v0,jump_intensity,jump_mean,jump_std)

    # Risk-neutral valuation formula
    call_price = (
        S0 * P1- K * np.exp(-r * T) * P2
    )

    # Put-call parity
    put_price = (call_price- S0+ K * np.exp(-r * T)
    )

    return call_price, put_price


if __name__ == "__main__":

    # Model parameters
    S0 = 100.0
    K = 100.0
    r = 0.05
    T = 1.0

    # Heston volatility parameters
    v0 = 0.04
    kappa = 1.5
    theta = 0.04
    sigma_v = 0.3
    rho = -0.5

    # Jump diffusion parameters
    jump_intensity = 0.1
    jump_mean = -0.05
    jump_std = 0.2

    # Compute Bates option prices
    call_price, put_price = bates_option_price(
        S0=S0,
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

    print(f"European Call Price: {call_price:.6f}")
    print(f"European Put Price : {put_price:.6f}")
