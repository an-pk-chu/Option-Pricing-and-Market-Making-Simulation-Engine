"""
market_maker.py
Simulates a single-underlying market maker who quotes a two-sided market
around a theoretical fair value, gets randomly "hit" by order flow, and
manages inventory risk by skewing quotes away from a target of zero.

This is a simplified version of the Avellaneda-Stoikov style market-making
framework: quote width depends on volatility and time horizon, and quote
midpoint is skewed by current inventory to mean-revert the position.
"""

import numpy as np


def simulate_gbm_path(S0, mu, sigma, T, n_steps, seed=None):
    """Simulate one geometric Brownian motion price path."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal(n_steps)
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    log_path = np.log(S0) + np.cumsum(log_returns)
    path = np.concatenate([[S0], np.exp(log_path)])
    return path


def run_market_maker_sim(
    S0=100.0,
    mu=0.0,
    sigma=0.25,
    T=1.0,               # trading horizon in years (e.g. 1.0 = one year of trading)
    n_steps=252,         # daily steps
    base_half_spread=0.10,   # base half-spread quoted around fair value
    vol_spread_coef=0.5,     # how much spread widens with realized vol
    inventory_skew_coef=0.02,  # how much quotes skew per unit of inventory
    inventory_limit=50,      # max absolute inventory before refusing further same-side fills
    fill_intensity=0.6,      # probability per step that a quote gets lifted/hit
    seed=42,
):
    """
    Runs a discrete-time market-making simulation and returns a dict of
    time series: prices, inventory, cash, mark-to-market PnL, quotes.

    Mechanics per step:
      1. Underlying moves one GBM step.
      2. MM sets bid/ask around current fair value, widened by realized vol
         and skewed by current inventory (long inventory -> skew quotes
         down to encourage selling / discourage buying more, and vice versa).
      3. Random order flow independently hits the bid and/or ask with
         probability `fill_intensity`, unless inventory limit blocks that side.
      4. Cash and inventory update on fills; PnL is marked to the new
         underlying price each step.
    """
    rng = np.random.default_rng(seed)
    prices = simulate_gbm_path(S0, mu, sigma, T, n_steps, seed=seed)

    inventory = np.zeros(n_steps + 1)
    cash = np.zeros(n_steps + 1)
    pnl = np.zeros(n_steps + 1)
    bids = np.zeros(n_steps + 1)
    asks = np.zeros(n_steps + 1)
    n_trades = 0

    # realized vol estimate, updated on a rolling window for spread-setting
    window = 20
    for t in range(1, n_steps + 1):
        ret_window = np.diff(np.log(prices[max(0, t - window):t + 1]))
        realized_vol = np.std(ret_window) * np.sqrt(252) if len(ret_window) > 2 else sigma

        fair_value = prices[t]
        half_spread = base_half_spread + vol_spread_coef * realized_vol
        skew = -inventory_skew_coef * inventory[t - 1]  # long inventory -> lower quotes

        bid = fair_value - half_spread + skew
        ask = fair_value + half_spread + skew
        bids[t], asks[t] = bid, ask

        inv = inventory[t - 1]
        csh = cash[t - 1]

        # Random flow: independent coin flips for "someone sells to our bid"
        # and "someone buys from our ask"
        if rng.random() < fill_intensity and inv < inventory_limit:
            inv += 1          # we buy at our bid
            csh -= bid
            n_trades += 1
        if rng.random() < fill_intensity and inv > -inventory_limit:
            inv -= 1          # we sell at our ask
            csh += ask
            n_trades += 1

        inventory[t] = inv
        cash[t] = csh
        pnl[t] = csh + inv * fair_value

    return {
        "prices": prices,
        "inventory": inventory,
        "cash": cash,
        "pnl": pnl,
        "bids": bids,
        "asks": asks,
        "n_trades": n_trades,
    }


def performance_stats(pnl, n_steps_per_year=252):
    """Compute Sharpe ratio, max drawdown, and total/annualized return stats
    for a PnL series (in dollar terms, not returns, since this is a
    market-making book rather than a return-on-capital strategy)."""
    daily_pnl = np.diff(pnl)
    mean_daily = daily_pnl.mean()
    std_daily = daily_pnl.std()

    sharpe = (mean_daily / std_daily) * np.sqrt(n_steps_per_year) if std_daily > 0 else np.nan

    running_max = np.maximum.accumulate(pnl)
    drawdown = pnl - running_max
    max_drawdown = drawdown.min()

    return {
        "total_pnl": pnl[-1],
        "mean_daily_pnl": mean_daily,
        "std_daily_pnl": std_daily,
        "sharpe_annualized": sharpe,
        "max_drawdown": max_drawdown,
    }


if __name__ == "__main__":
    result = run_market_maker_sim()
    stats = performance_stats(result["pnl"])
    print(f"Trades executed: {result['n_trades']}")
    print(f"Final inventory: {result['inventory'][-1]:.0f}")
    print(f"Total PnL: ${stats['total_pnl']:.2f}")
    print(f"Annualized Sharpe: {stats['sharpe_annualized']:.2f}")
    print(f"Max drawdown: ${stats['max_drawdown']:.2f}")
