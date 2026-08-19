"""
pricing.py
Analytical and semi-analytical option pricing tools.

Implements:
  - Black-Scholes-Merton closed-form price + full Greeks (delta, gamma, vega, theta, rho)
  - Newton-Raphson implied volatility solver with bisection fallback
  - Cox-Ross-Rubinstein binomial tree for American-style options (early exercise)

All functions are vectorized with numpy where practical so they can be
called across a whole option chain at once.
"""

import numpy as np
from scipy.stats import norm


def bs_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Black-Scholes-Merton price for a European option.

    S : spot price
    K : strike
    T : time to expiry in years
    r : risk-free rate (continuously compounded)
    sigma : volatility (annualized)
    q : continuous dividend yield
    """
    if T <= 0:
        # At expiry, price collapses to intrinsic value
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price


def bs_greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Returns a dict of Greeks: delta, gamma, vega, theta (per year), rho.
    Vega and gamma are identical for calls/puts; delta, theta, rho differ.
    """
    if T <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    pdf_d1 = norm.pdf(d1)

    gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T)  # per 1.0 change in sigma (i.e. /100 for 1 vol pt)

    if option_type == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (
            -S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1)
        )
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (
            -S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
        )
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(price, S, K, T, r, option_type="call", q=0.0,
                 tol=1e-8, max_iter=100, sigma_init=0.3):
    """
    Solve for implied volatility given an observed market price, using
    Newton-Raphson (fast, uses vega) with a bisection fallback if Newton
    fails to converge or vega collapses near expiry / deep ITM-OTM.
    """
    sigma = sigma_init
    for _ in range(max_iter):
        model_price = bs_price(S, K, T, r, sigma, option_type, q)
        vega = bs_greeks(S, K, T, r, sigma, option_type, q)["vega"]
        diff = model_price - price

        if abs(diff) < tol:
            return sigma
        if vega < 1e-8:
            break  # vega too small, Newton step is unstable -> fall back
        sigma -= diff / vega
        if sigma <= 0:
            sigma = 1e-4

    # --- Bisection fallback ---
    lo, hi = 1e-4, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        model_price = bs_price(S, K, T, r, mid, option_type, q)
        if model_price > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def binomial_american(S, K, T, r, sigma, option_type="put", q=0.0, n_steps=500):
    """
    Cox-Ross-Rubinstein binomial tree price for an American option.
    American puts are the classic case where early exercise has real value
    (e.g. deep ITM put on a non-dividend stock), which Black-Scholes cannot
    capture since it only prices European exercise.
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)

    # Terminal stock prices
    j = np.arange(n_steps + 1)
    ST = S * (u ** (n_steps - j)) * (d ** j)

    if option_type == "call":
        values = np.maximum(ST - K, 0.0)
    else:
        values = np.maximum(K - ST, 0.0)

    # Backward induction with early-exercise check at every node
    for step in range(n_steps - 1, -1, -1):
        j = np.arange(step + 1)
        ST = S * (u ** (step - j)) * (d ** j)
        continuation = disc * (p * values[:-1] + (1 - p) * values[1:])
        if option_type == "call":
            exercise = np.maximum(ST - K, 0.0)
        else:
            exercise = np.maximum(K - ST, 0.0)
        values = np.maximum(continuation, exercise)

    return values[0]


if __name__ == "__main__":
    # Sanity check against known reference values
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    call = bs_price(S, K, T, r, sigma, "call")
    put = bs_price(S, K, T, r, sigma, "put")
    print(f"ATM 1y call: {call:.4f}  (expected ~10.4506)")
    print(f"ATM 1y put:  {put:.4f}  (expected ~5.5735)")

    # Put-call parity check: C - P = S*e^-qT - K*e^-rT
    parity_lhs = call - put
    parity_rhs = S - K * np.exp(-r * T)
    print(f"Put-call parity check: {parity_lhs:.6f} vs {parity_rhs:.6f}")

    iv = implied_vol(call, S, K, T, r, "call")
    print(f"Recovered implied vol: {iv:.6f} (should be 0.20)")

    amer_put = binomial_american(S, K, T, r, sigma, "put")
    euro_put = bs_price(S, K, T, r, sigma, "put")
    print(f"American put: {amer_put:.4f}  European put: {euro_put:.4f}  "
          f"(early exercise premium: {amer_put - euro_put:.4f})")
