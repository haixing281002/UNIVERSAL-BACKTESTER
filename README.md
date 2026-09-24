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
| `allocators.py` | `equal_weight`, `cross_sectional` (rank/select/weight), `time_series_momentum`, `atr_risk_parity` |
| `signals.py` | Rolling-regression momentum score, SMA, realized vol, ATR, gap-move flag, regime filter — all raw/unshifted, by design |
| `data.py` | `load_wide_csv`, `load_banner_workbook` / `load_field_only` (multi-series Excel exports), `audit_frame` (data-quality checks) |
| `metrics.py` | CAGR, vol, Sharpe, max drawdown, turnover |
| `validation.py` | Block-bootstrap Sharpe CI, deflated Sharpe ratio — is a result skill or luck |
| `tearsheet.py` | 27-metric risk/return tearsheet: vol split, drawdown, Sharpe/Treynor/Sortino/Calmar/Sterling/Omega, capture ratios, VaR/CVaR, skew/kurtosis — see below |
| `reporting.py` | `derive_trade_log` — reconstructs discrete BUY/SELL events from the daily weight path |
| `examples/momentum_rotation.py` | A full worked cross-sectional momentum strategy, generic over whatever price file you point it at |
| `tests/` | 40 tests: causality + survivorship (`test_causality.py`), weekday schedule + regime gate (`test_regime_and_schedule.py`), bootstrap/deflated Sharpe (`test_validation.py`), all 27 tearsheet metrics (`test_tearsheet.py`) |

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

## Where to see what it bought and sold

Running the example writes a full buy/sell log to `trade_log.csv` (repo
root), one row per `(date, asset)` where the position changed on a rebalance
date -- action (BUY/SELL), weight before, weight after. It's derived from
the engine's daily weight path via `universal_backtester.derive_trade_log`,
which you can call on any `BacktestResult` the same way. This is a
**weight-based** engine, not a shares-and-cash ledger: "buying an index"
here means allocating a fraction of NAV to a notional unit that tracks that
index's price 1:1 (no tracking error, no expense ratio, no minimum lot
size) -- see "Design boundary" below for what that does and doesn't let you
claim. The strategy is **long-only with no leverage**: `engine.py` clips
every target weight at `>= 0` and caps total exposure at 100%, structurally
-- there is no parameter that allows shorting or borrowing.

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

## How closely the example follows Clenow's actual rules

| Rule (the paper's forensic extract) | This example |
|---|---|
| 90-day exp-regression slope × R² ranking | Matches |
| Above 100-day MA to qualify | Matches |
| Disqualify >15% move in trailing 90 days | Matches |
| Trade on Wednesday | Matches (`rebalance="weekly_wed"` — falls back to the nearest trading day on a Wednesday holiday) |
| Hold the top 20% of the ranking | Matches (`quantile=0.20`) |
| Size by `AccountValue × risk_factor / ATR20` (risk parity) | Matches (`atr_risk_parity` allocator, real High/Low from the bundled feed) |
| New buys blocked when benchmark < 200-DMA; existing names held, not force-sold | Matches (`regime=` → `ctx.buys_allowed`) |
| No stop-loss | Matches (trivially — neither has one) |
| ~500 individual S&P 500 stocks | **Does not match** — 10 Indian index series instead, no stock data exists here |
| Biweekly separate resize step | Approximated — recomputes the full ATR target every Wednesday instead of every second one (more frequent, not less) |
| Checks/trades ONLY on Wednesday | Minor deviation — a 100-DMA/gap sell trigger can act on the first day it's causally knowable, which need not be a Wednesday |

Passing this rule set through the real engine, instead of equal-weighting
with no regime filter (an earlier, looser version of this example), changes
the headline number a lot: CAGR drops from ~14%/year to ~2.8%/year, because
ATR risk-parity sizing on a smooth, diversified index (low ATR relative to
price) produces heavy, real cash drag — ~86% of NAV sits in cash on average.
That's not a bug to fix; it's what the paper's own sizing formula does when
translated onto this instrument set, and it's worth sitting with before
concluding anything about whether the idea "works" here.

## The full tearsheet

`examples/momentum_rotation.py` prints a 27-metric tearsheet (`universal_backtester.tearsheet`) whenever it has a designated benchmark column (the bundled feed's NIFTY 500):
volatility split (annualized, positive/negative), max drawdown, average annual max drawdown, Sharpe, Treynor, Sortino, Calmar, Sterling, Omega, Gain-to-Pain, Tail Ratio, Alpha (Bloomberg-adjusted beta), Tracking Error, conditional returns on up/down benchmark days, Upside/Downside/Capture/Extreme Capture ratios, skewness, kurtosis, and 95%/99% VaR and CVaR.

**Read this before trusting any benchmark-relative number in it.** Every NIFTY 500 series in this repo is a *price-return* index — dividends excluded — used as a stand-in for the *Total Return Index* that "vs. Nifty 500" normally means. A price-return benchmark is a systematically weaker comparison, understating the true benchmark by roughly its dividend yield (~1.3–1.5%/year for Indian large/mid caps). Alpha, tracking error, capture ratios and the conditional-return pair are all computed against that weaker benchmark, which flatters the strategy relative to a true TRI comparison by about that much. They're fine for comparing two runs against the *same* proxy; not fine as a standalone "beat the index by X" claim.

Capture ratios compound **monthly** returns of the up/down subset, matching standard (Morningstar-style) practice — compounding a non-contiguous subset of *daily* returns back-to-back distorts the number by an order of magnitude or more (confirmed while building this: a naive daily version turned a ~150%-expected capture ratio into ~2400%). Even at monthly granularity, compounding is convex in a scale factor — a portfolio at a clean 1.5x the benchmark every month will show capture ratios noticeably *above* 150%, not exactly 150% — a real property of the metric, not a bug; see the tests in `test_tearsheet.py` for the worked proof.

## Tests

```bash
pytest
```
