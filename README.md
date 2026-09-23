# universal-backtester

A small, dependency-light, **causally-correct** long-only backtesting engine.
Not tied to any fund, market, or data source — point it at any
`DatetimeIndex x asset` price frame and it will run a strategy without
letting a signal trade the bar that produced it.

This exists because a backtest that quietly lets a signal see the future, or
quietly drops the names that went to zero, produces a number that looks
exactly like a real result until someone checks. The engine makes both
mistakes structurally hard:

- **Causality.** Every signal (alpha score, eligibility, volatility) is
  shifted by `1 + lag_days` before an allocator can see it. The `+1` removes
  same-bar use of the close that produced the signal; `lag_days` is any
  extra implementation delay you declare (e.g. an index that publishes after
  the close). There is no parameter that gets you same-bar execution.
- **No free survivorship.** A fixed asset set must have complete price data
  — a NaN is treated as a data fault, not silently skipped. A *changing*
  universe (names listing/delisting) must be declared explicitly via
  `membership=`, and a name that leaves membership is **sold at cost** the
  day it's learned, never quietly zeroed and the rest renormalised for free.
- **Costs are charged on every trade**, including forced exits, using a
  half-spread model driven by `spread_bps`.
- **`assert_causal()`** is a cheap tripwire you can run on any signal before
  trusting it: it checks correlation between `signal[t]` and `return[t+h]`
  for `h in {0, 1}` and raises if a signal looks like it knows the future.

## Install

```bash
git clone https://github.com/haixing281002/universal-backtester.git
cd universal-backtester
pip install -e ".[dev]"
```

## Quickstart

```python
import pandas as pd
from universal_backtester import Backtester, build_allocator
from universal_backtester.metrics import compute_metrics

# prices: DatetimeIndex x asset, one column per tradable name
# alpha:  same shape, your ranking/expected-return score (RAW, unshifted --
#         the engine shifts it for you)
bt = Backtester(prices=prices, assets=list(prices.columns),
                spread_bps=30, lag_days=1, allow_cash=True)

alloc = build_allocator("cross_sectional", list(prices.columns),
                        quantile=0.2, weighting="equal", min_names=5)

result = bt.run(allocator=alloc, rebalance="weekly", alpha=alpha, warmup=90)
print(compute_metrics(result))
```

## What's in the box

| Module | What it does |
|---|---|
| `engine.py` | `Backtester` — the causal accounting loop, `assert_causal()` tripwire |
| `allocators.py` | `equal_weight`, `cross_sectional` (rank/select/weight), `time_series_momentum` |
| `signals.py` | Rolling-regression momentum score, SMA, realized vol, gap-move flag — all raw/unshifted, by design |
| `data.py` | `load_wide_csv`, `load_banner_workbook` (multi-series Excel exports), `audit_frame` (data-quality checks) |
| `metrics.py` | CAGR, vol, Sharpe, max drawdown, turnover |
| `examples/momentum_rotation.py` | A full worked cross-sectional momentum strategy, generic over whatever price file you point it at |
| `tests/test_causality.py` | Proves the engine's causal guarantee and survivorship handling — plants the bias, confirms it's caught |

## Design boundary — what this deliberately does NOT do

This is a **generic engine**, not a fund's governed research pipeline. It has
no notion of a Strategy Card, an approval gate, a specific market's trading
non-negotiables (settlement lag, cost floors, backfill caveats), or a
declared data registry. If you're adapting a specific paper's strategy for a
specific mandate, wrap this engine in whatever process fits that mandate —
don't expect it to enforce one for you.

## Run the example

```bash
python examples/momentum_rotation.py                     # uses the bundled data feed below
python examples/momentum_rotation.py path/to/your_prices.xlsx   # or point it at your own
```

The example implements a Clenow "Stocks on the Move"-style ranking (90-day
exponential-regression slope × R², 100-day trend filter, 15% gap filter) as
a cross-sectional rotation across whatever columns your price file has. It
prints metrics for both the causally-correct run and, for comparison, a
deliberately-broken same-bar replay — so you can see the size of the leak on
your own data before trusting any number.

## Bundled data feed (hardcoded default, as of now)

`data/Indices_Historical_Data.xlsx` is checked into this repo and is the
default the example runs against when called with no arguments. Two sheets:

- **Broad Market** — NIFTY cap-segment indices (50 / 100 / 200 / 500 /
  Midcap 100 / Midcap 150 / Smallcap 100 / Smallcap 250 / Midsmallcap 400 /
  Microcap 250)
- **Factor Indices** — NIFTY smart-beta sleeves (Alpha 50, 500-Momentum 50,
  Multifactor MQVLV 50, 500-Quality 50, 500-Value 50, 500-Low-Vol 50,
  High-Beta 50, 200-Value 30, 200-Momentum 30, 200-Quality 30)

The example trades a curated, non-redundant 10-index subset of these (see
`UNIVERSE` in `examples/momentum_rotation.py`) against NIFTY 500 as
benchmark/regime reference — **these are index series, not individual
stocks**, so this is a sleeve-rotation strategy, not a stock-picker, however
the ranking logic is dressed up. Swap in your own file any time; nothing in
the engine or allocators is tied to this data.

## Tests

```bash
pytest
```
