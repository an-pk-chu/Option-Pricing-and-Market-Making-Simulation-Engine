# Options Pricing & Market-Making Simulation Engine

A from-scratch Python engine covering the two core disciplines of prop
options trading: **pricing derivatives correctly** and **managing risk while
making markets in them**. Built to be interview-defensible — every number
is reproducible, every model is cross-validated against an independent
method, and every design choice has a stated tradeoff.

## What it does

**1. Derivatives pricing (`pricing.py`)**
- Black-Scholes-Merton closed-form pricer with full Greeks (delta, gamma,
  vega, theta, rho), supporting dividend yield
- Newton-Raphson implied volatility solver with a bisection fallback for
  when vega collapses (deep ITM/OTM, near expiry)
- Cox-Ross-Rubinstein binomial tree for American-style options, to capture
  early-exercise value that Black-Scholes structurally cannot price

**2. Independent cross-validation (`backtest.py`)**
- A second, independent Monte Carlo pricer (with antithetic variates for
  variance reduction) verifies the Black-Scholes price to within a few
  tenths of a cent — good practice before trusting a pricing model with
  real risk capital
- Confirms the American put trades at a premium to the European put via
  the binomial tree, isolating the early-exercise value

**3. Market-making simulation (`market_maker.py`)**
- Simulates a GBM underlying and a market maker quoting a two-sided market
  around fair value
- Quote width scales with realized volatility (wider quotes when the
  market is choppier)
- Quotes skew away from a flat inventory target — long inventory pushes
  quotes down to encourage selling, short inventory pushes them up — the
  same first-order idea behind Avellaneda-Stoikov-style inventory control
- Tracks cash, inventory, and mark-to-market PnL tick by tick

**4. Monte Carlo ensemble backtest (`backtest.py`)**
- A single simulated path tells you almost nothing about a strategy — this
  runs the market maker across 500 independent price paths and reports the
  *distribution* of outcomes: mean PnL, win rate, Sharpe ratio, drawdown
- Runs a spread-sensitivity sweep showing the central market-making
  tradeoff: wider quoted spreads increase edge captured per trade and
  Sharpe ratio, at the cost of higher variance and lower fill rate

## Results (base case: $100 spot, 25% vol, $0.10 base half-spread, 252 trading days)

| Metric | Value |
|---|---|
| Mean total PnL (500 paths) | +$65.86 |
| Win rate | 70.2% |
| Mean annualized Sharpe | 0.54 |
| Mean max drawdown | -$178.84 |
| Worst-case drawdown (500 paths) | -$1,063.26 |

Widening the quoted half-spread from $0.05 to $0.40 roughly triples mean
PnL and takes mean Sharpe from 0.43 to 1.29 — the expected result, since a
wider spread captures more edge per round-trip. In a real book this
tradeoff is bounded by adverse selection (wider quotes get picked off less
often by informed flow, but also win less flow overall) and by competition
from other market makers, neither of which this simplified model captures.
See "Known simplifications" below.

## Known simplifications (what I'd add next)

This is an honest v1, not a claim of a production strategy:
- Order flow is modeled as i.i.d. random fills rather than a realistic
  limit order book with adverse selection (informed traders disproportionately
  hitting the correct side of the quote)
- No transaction costs, exchange fees, or latency modeling
- Volatility is estimated from a simple rolling realized-vol window rather
  than a proper GARCH or implied-vol-surface model
- Inventory skew coefficient is fixed rather than solved for optimally
  (the real Avellaneda-Stoikov framework derives the optimal skew from a
  utility-maximization problem — that's the natural next step)

## How to run

```bash
python pricing.py      # sanity checks: put-call parity, IV recovery, American vs European
python market_maker.py # single simulated path, one seed
python backtest.py     # full pipeline: cross-validation + 500-path ensemble + spread sweep + chart
```

Requires: `numpy`, `scipy`, `matplotlib`.
