"""
backtest.py
A single simulated path tells you almost nothing about a trading strategy -
you need the distribution of outcomes. This module runs the market maker
across many independent price paths (Monte Carlo ensemble) and reports:

  - Distribution of total PnL, win rate, Sharpe ratio, max drawdown
  - A parameter sweep over spread width to show the classic market-making
    tradeoff: wider spreads reduce fill rate but increase edge per trade

Also cross-checks the analytic option pricer against the Monte Carlo /
binomial pricers as an independent verification of correctness, which
is good practice before trusting a pricing model with real risk.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from market_maker import run_market_maker_sim, performance_stats
from pricing import bs_price, binomial_american


def monte_carlo_option_price(S, K, T, r, sigma, option_type="call",
                              n_paths=200_000, seed=0):
    """
    Independent Monte Carlo pricer for a European option using antithetic
    variates for variance reduction. Used here purely to cross-validate
    the closed-form Black-Scholes price - if the two disagree by more than
    a couple cents, something is wrong with one of the implementations.
    """
    rng = np.random.default_rng(seed)
    half = n_paths // 2
    z = rng.standard_normal(half)
    z = np.concatenate([z, -z])  # antithetic pairing halves variance

    ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    discounted = np.exp(-r * T) * payoff
    price = discounted.mean()
    stderr = discounted.std() / np.sqrt(n_paths)
    return price, stderr


def cross_validate_pricers():
    S, K, T, r, sigma = 100, 105, 0.5, 0.04, 0.28
    bs = bs_price(S, K, T, r, sigma, "call")
    mc_price, mc_se = monte_carlo_option_price(S, K, T, r, sigma, "call")
    amer_put_binom = binomial_american(S, K, T, r, sigma, "put")
    euro_put_bs = bs_price(S, K, T, r, sigma, "put")

    print("=== Pricer Cross-Validation ===")
    print(f"Black-Scholes call:      {bs:.4f}")
    print(f"Monte Carlo call:        {mc_price:.4f}  (+/- {1.96*mc_se:.4f}, 95% CI)")
    print(f"Difference:              {abs(bs - mc_price):.4f}")
    print(f"American put (binomial): {amer_put_binom:.4f}")
    print(f"European put (BS):       {euro_put_bs:.4f}")
    print(f"Early-exercise premium:  {amer_put_binom - euro_put_bs:.4f}\n")


def run_ensemble(n_sims=500, **mm_kwargs):
    """Run the market-making strategy across n_sims independent price paths."""
    records = []
    for seed in range(n_sims):
        result = run_market_maker_sim(seed=seed, **mm_kwargs)
        stats = performance_stats(result["pnl"])
        stats["n_trades"] = result["n_trades"]
        stats["final_inventory"] = result["inventory"][-1]
        records.append(stats)
    return records


def summarize_ensemble(records, label=""):
    pnls = np.array([r["total_pnl"] for r in records])
    sharpes = np.array([r["sharpe_annualized"] for r in records])
    drawdowns = np.array([r["max_drawdown"] for r in records])

    print(f"=== Ensemble Summary {label} (n={len(records)} paths) ===")
    print(f"Mean total PnL:        ${pnls.mean():.2f}")
    print(f"Std dev of PnL:        ${pnls.std():.2f}")
    print(f"Win rate (PnL>0):      {(pnls > 0).mean()*100:.1f}%")
    print(f"Mean annualized Sharpe:{np.nanmean(sharpes):.2f}")
    print(f"Mean max drawdown:     ${drawdowns.mean():.2f}")
    print(f"Worst-case drawdown:   ${drawdowns.min():.2f}\n")
    return pnls, sharpes, drawdowns


def spread_sensitivity_sweep(spreads=(0.05, 0.10, 0.15, 0.20, 0.30, 0.40)):
    """
    Classic market-making tradeoff: as quoted spread widens, edge per trade
    goes up but fill rate goes down (modeled here as fewer trades happening
    to be economically rational, approximated by reduced fill intensity as
    a stand-in for a smarter agent-based counterparty). We hold fill
    intensity constant here and instead show how PnL and Sharpe evolve
    purely from wider capture per round-trip vs. increased inventory risk.
    """
    results = []
    for spread in spreads:
        records = run_ensemble(n_sims=300, base_half_spread=spread)
        pnls = np.array([r["total_pnl"] for r in records])
        sharpes = np.array([r["sharpe_annualized"] for r in records])
        results.append({
            "spread": spread,
            "mean_pnl": pnls.mean(),
            "std_pnl": pnls.std(),
            "mean_sharpe": np.nanmean(sharpes),
        })
    return results


def plot_results(pnls, sharpes, sweep_results, out_path="ensemble_results.png"):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].hist(pnls, bins=30, color="#2c6e91", edgecolor="white")
    axes[0, 0].axvline(0, color="black", linewidth=1, linestyle="--")
    axes[0, 0].set_title("Distribution of Total PnL Across 500 Simulated Paths")
    axes[0, 0].set_xlabel("Total PnL ($)")
    axes[0, 0].set_ylabel("Frequency")

    axes[0, 1].hist(sharpes[~np.isnan(sharpes)], bins=30, color="#a13d3d", edgecolor="white")
    axes[0, 1].axvline(0, color="black", linewidth=1, linestyle="--")
    axes[0, 1].set_title("Distribution of Annualized Sharpe Ratio")
    axes[0, 1].set_xlabel("Sharpe Ratio")
    axes[0, 1].set_ylabel("Frequency")

    spreads = [r["spread"] for r in sweep_results]
    mean_pnl = [r["mean_pnl"] for r in sweep_results]
    std_pnl = [r["std_pnl"] for r in sweep_results]
    axes[1, 0].errorbar(spreads, mean_pnl, yerr=std_pnl, fmt="o-", color="#2c6e91", capsize=4)
    axes[1, 0].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[1, 0].set_title("Mean PnL vs. Quoted Half-Spread (+/- 1 std dev)")
    axes[1, 0].set_xlabel("Base Half-Spread ($)")
    axes[1, 0].set_ylabel("Mean Total PnL ($)")

    mean_sharpe = [r["mean_sharpe"] for r in sweep_results]
    axes[1, 1].plot(spreads, mean_sharpe, "o-", color="#a13d3d")
    axes[1, 1].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[1, 1].set_title("Mean Sharpe Ratio vs. Quoted Half-Spread")
    axes[1, 1].set_xlabel("Base Half-Spread ($)")
    axes[1, 1].set_ylabel("Mean Annualized Sharpe")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved chart to {out_path}")


if __name__ == "__main__":
    cross_validate_pricers()

    records = run_ensemble(n_sims=500)
    pnls, sharpes, drawdowns = summarize_ensemble(records, label="(base case)")

    sweep = spread_sensitivity_sweep()
    print("=== Spread Sensitivity Sweep ===")
    for r in sweep:
        print(f"half_spread=${r['spread']:.2f}  mean_pnl=${r['mean_pnl']:.2f}  "
              f"std_pnl=${r['std_pnl']:.2f}  mean_sharpe={r['mean_sharpe']:.2f}")

    plot_results(pnls, sharpes, sweep, out_path="ensemble_results.png")
