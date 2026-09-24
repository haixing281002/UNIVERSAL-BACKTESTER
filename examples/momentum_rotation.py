"""Worked example: a Clenow "Stocks on the Move"-style cross-sectional
momentum rotation, built on the generic universal_backtester engine.

DATA FEED (hardcoded, as of now): data/Indices_Historical_Data.xlsx, bundled
in this repo. It has two sheets -- "Broad Market" (NIFTY cap-segment
indices: 50/100/200/500/Midcap/Smallcap/Microcap) and "Factor Indices" (NIFTY
Alpha/Momentum/Quality/Value/Low-Vol/High-Beta/200-variant sleeves), each
with full Open/High/Low/Close/PE/PB fields. Nothing here is stock-level: the
rotation universe is a curated, non-redundant subset of these ~20 index
series (see UNIVERSE below), not individual securities. That is a real,
load-bearing constraint on what this example can claim -- see README.md's
"Design boundary" section.

HOW CLOSELY THIS FOLLOWS CLENOW'S ACTUAL RULES (the book's forensic extract):
  - 90-day exponential-regression slope x R^2 ranking            MATCHES
  - Above 100-day MA to qualify                                  MATCHES
  - Disqualify >15% move in trailing 90 days                     MATCHES
  - Trade on WEDNESDAY (previously wrong -- was landing on Friday
    via a generic "weekly" schedule; fixed via rebalance="weekly_wed") MATCHES
  - Hold the TOP 20% of the ranking (was 30% in an earlier version)   MATCHES
  - Size by AccountValue*risk_factor/ATR20 (risk parity, not equal
    weight -- via the "atr_risk_parity" allocator, using real
    High/Low from the bundled feed)                               MATCHES
  - New buys blocked when benchmark < 200-DMA; existing names held,
    not force-sold, just not replaced (via `regime=` / buys_allowed)  MATCHES
  - No stop-loss                                                  MATCHES (neither has one)
  - Universe = ~500 individual S&P 500 stocks                     DOES NOT MATCH
    (ranks 10 Indian INDEX series instead -- no stock data exists here)
  - Biweekly separate resize step                                 APPROXIMATED
    (atr_risk_parity recomputes the full ATR-based target every
    Wednesday rather than only every second one -- MORE frequent
    resizing than the book, not less; documented, not hidden)
  - Sell triggers (100-DMA/gap) act on the day they're causally
    knowable, which can be any day, not only Wednesday             MINOR DEVIATION
    (the book only checks/trades on Wednesdays; this engine's
    membership-driven forced-exit can fire mid-week -- conservative,
    not a source of flattery, but still not literally "Wednesday only")

Usage:
    python examples/momentum_rotation.py                      # uses the bundled feed
    python examples/momentum_rotation.py path/to/other.xlsx    # override (simpler, no ATR/regime -- see below)
    python examples/momentum_rotation.py path/to/other.csv

An override file only gets the ranking + filter rules (no ATR sizing, no
regime filter) unless it has the same Broad-Market/Factor-Indices shape with
real High/Low columns -- ATR needs real High/Low, and a regime filter needs
a designated benchmark column, neither of which a generic CSV declares.

It runs the SAME strategy two ways and prints both:
  1. correctly, through the engine (signal shifted 1+lag_days before use)
  2. a deliberately-broken same-bar replay (decide and trade off the same
     close) -- to show you, on your own data, how large that leak is.
Never trust (2); it exists only to quantify the mistake.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__file__.rsplit("/", 2)[0]))  # allow running without `pip install -e .`

from universal_backtester import Backtester, build_allocator, assert_causal, LookaheadError, derive_trade_log
from universal_backtester.data import load_wide_csv, load_banner_workbook, load_field_only
from universal_backtester.metrics import metrics_table, render_table
from universal_backtester.signals import rolling_regression_momentum, sma, max_abs_move_flag, atr, regime_filter
from universal_backtester.validation import bootstrap_sharpe_ci, deflated_sharpe_from_returns
from universal_backtester.tearsheet import compute_tearsheet, render_tearsheet

REG_WINDOW = 90
SMA_WINDOW = 100          # 100-day trend qualifier
SMA_REGIME = 200          # 200-day benchmark regime filter
GAP_WINDOW = 90
GAP_THRESHOLD = 0.15
ATR_WINDOW = 20
RISK_FACTOR = 0.001       # Clenow's target daily $ impact per position = 0.1% of NAV
SPREAD_BPS = 30.0         # not specified by the paper; a documented assumption
LAG_DAYS = 1
TOP_QUANTILE = 0.20       # "top 20% of the ranking"
MIN_NAMES = 3
WARMUP = max(REG_WINDOW, SMA_REGIME) + 5

DEFAULT_DATA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "Indices_Historical_Data.xlsx")

# Curated, non-redundant rotation universe out of the ~20 index series the
# bundled feed has. Dropped: NIFTY 100/200 (near-duplicates of 50/500),
# MIDCAP 100 / SMALLCAP 100 / MIDSMALLCAP 400 (redundant with the 150/250-name
# versions kept below), and NIFTY200 VALUE/MOMENTUM/QUALITY 30 (redundant
# with the NIFTY500-scoped factor sleeves kept below, and narrower).
UNIVERSE = [
    "NIFTY 50", "NIFTY MIDCAP 150", "NIFTY SMALLCAP 250", "NIFTY MICROCAP 250",
    "NIFTY ALPHA 50", "NIFTY500 MOMENTUM 50", "NIFTY500 MULTIFACTOR MQVLV 50",
    "NIFTY500 QUALITY 50", "NIFTY500 VALUE 50", "NIFTY500 LOW VOLATILITY 50",
]
BENCHMARK = "NIFTY 500"   # regime filter reference + comparison, not traded


def load_default_feed():
    """Close, High, Low for UNIVERSE + BENCHMARK from the bundled feed."""
    cols = UNIVERSE + [BENCHMARK]
    close, _ = load_field_only(DEFAULT_DATA_FILE, field="Close")
    high, _ = load_field_only(DEFAULT_DATA_FILE, field="High")
    low, _ = load_field_only(DEFAULT_DATA_FILE, field="Low")
    close = close[[c for c in cols if c in close.columns]].dropna()
    high = high.reindex(close.index)[close.columns]
    low = low.reindex(close.index)[close.columns]
    # A handful of older rows have High/Low missing ('-' dashes) while Close
    # is populated; fall back to Close so True Range degrades gracefully to
    # |Close - PrevClose| rather than producing a gap in the ATR.
    high = high.fillna(close)
    low = low.fillna(close)
    return close, high, low


def load_override(path: str, sheet: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        df, _ = load_banner_workbook(path, sheet=sheet)
    else:
        df, _ = load_wide_csv(path)
    return df.select_dtypes(include=[float, int]).dropna(how="all").dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("price_file", nargs="?", default=DEFAULT_DATA_FILE)
    ap.add_argument("--sheet", default="Sheet1")
    args = ap.parse_args()
    using_default = os.path.abspath(args.price_file) == DEFAULT_DATA_FILE

    if using_default:
        close_full, high, low = load_default_feed()
        close = close_full.drop(columns=[BENCHMARK])
        benchmark_col = BENCHMARK
        have_atr = True
    else:
        close_full = load_override(args.price_file, args.sheet)
        close = close_full
        high = low = None
        benchmark_col = None
        have_atr = False

    assets = list(close.columns)
    print(f"Data feed: {args.price_file}{'  (bundled default)' if using_default else ''}")
    print(f"Loaded {len(assets)} tradable series"
          + (f" + benchmark '{benchmark_col}'" if benchmark_col else "")
          + f", {close.index.min().date()} -> {close.index.max().date()} ({len(close)} rows)"
          f"\nAssets: {assets}")
    if not have_atr:
        print("(override file: no High/Low columns declared -> ATR sizing and the "
              "regime filter are skipped; ranking + 100-DMA + gap filter still apply)")

    adj_slope = pd.DataFrame(index=close.index, columns=assets, dtype=float)
    eligible = pd.DataFrame(index=close.index, columns=assets, dtype=bool)
    atr_pct = pd.DataFrame(index=close.index, columns=assets, dtype=float)
    for name in assets:
        s, _ = rolling_regression_momentum(close[name], window=REG_WINDOW)
        adj_slope[name] = s
        above = close[name] > sma(close[name], SMA_WINDOW)
        gap = max_abs_move_flag(close[name], GAP_WINDOW, GAP_THRESHOLD)
        eligible[name] = above & (~gap)
        if have_atr:
            atr_pct[name] = atr(high[name], low[name], close[name], ATR_WINDOW) / close[name]

    regime = regime_filter(close_full[benchmark_col], SMA_REGIME) if benchmark_col else None

    returns_1d = close.pct_change()
    try:
        assert_causal(adj_slope, returns_1d, tol=0.35, label="adj_slope_raw", horizons=(0,))
        print("\n[causality] raw signal did not trip the tripwire at h=0 "
              "(a 90-day regression is not very sensitive to one day -- this "
              "does NOT mean same-bar execution is safe, see the replay below)")
    except LookaheadError as e:
        print(f"\n[causality] RAW SIGNAL TRIPPED THE TRIPWIRE:\n  {e}")

    bt = Backtester(prices=close, assets=assets, spread_bps=SPREAD_BPS,
                    lag_days=LAG_DAYS, allow_cash=True, membership=eligible)
    if have_atr:
        alloc = build_allocator("atr_risk_parity", assets, quantile=TOP_QUANTILE,
                                risk_factor=RISK_FACTOR, min_names=MIN_NAMES, ascending=False)
        result = bt.run(allocator=alloc, rebalance="weekly_wed", alpha=adj_slope, vols=atr_pct,
                        regime=regime, name="engine_correct", warmup=WARMUP)
    else:
        alloc = build_allocator("cross_sectional", assets, quantile=TOP_QUANTILE,
                                weighting="equal", min_names=MIN_NAMES, ascending=False)
        result = bt.run(allocator=alloc, rebalance="weekly_wed", alpha=adj_slope,
                        name="engine_correct", warmup=REG_WINDOW)

    tbl = metrics_table([result])
    print("\n" + "=" * 70)
    print(f"CORRECT ({'ATR risk-parity sizing, ' if have_atr else ''}"
          f"{'regime-gated, ' if benchmark_col else ''}causally-shifted, {SPREAD_BPS:.0f}bp cost,")
    print("LONG-ONLY -- no shorting, no leverage: the engine clips every target weight")
    print("at >= 0 and caps total exposure at 100%; unallocated NAV sits in cash)")
    print("=" * 70)
    print(render_table(tbl))
    print(f"mean cash weight: {result.cash_weight.mean():.1%}")
    if have_atr:
        diag = result.meta.get("allocator_diagnostics", {})
        print(f"mean names held: {diag.get('mean_names_held', float('nan')):.2f}  "
              f"| mean cash shortfall from ATR sizing alone: "
              f"{diag.get('mean_cash_shortfall_from_sizing', float('nan')):.1%}")
    if benchmark_col:
        print(f"fraction of history with new buys allowed (benchmark > {SMA_REGIME}-DMA): "
              f"{result.meta.get('mean_buys_allowed', float('nan')):.1%}")

    print("\n" + "-" * 70)
    print("STATISTICAL RIGOR -- is this Sharpe ratio trustworthy, or noise?")
    print("-" * 70)
    boot = bootstrap_sharpe_ci(result.returns, block_size=20, n_resamples=1000, seed=0)
    print(f"Sharpe ratio, 90% block-bootstrap CI: [{boot.ci_low:.2f}, {boot.ci_high:.2f}] "
          f"(point estimate {boot.point_estimate:.2f}, {boot.fraction_positive:.0%} of "
          f"resamples positive)")
    # N_TRIALS here is a rough, human-counted tally of the parameter/design
    # choices actually varied while building this example (universe size:
    # 7/10/20 names; weighting: equal/inverse-vol/ATR; top quantile:
    # 20%/30%; rebalance: weekly/weekly_wed; regime gate: on/off; cost:
    # 10bp/30bp) -- NOT a rigorously logged trial count the way BACKTESTER's
    # own trials_for_family() tracks it. Treat this DSR as illustrative:
    # the honest number requires logging every variation as you try it,
    # which this script does not yet do.
    N_TRIALS_ROUGH_ESTIMATE = 10
    dsr = deflated_sharpe_from_returns(result.returns, n_trials=N_TRIALS_ROUGH_ESTIMATE,
                                       trial_sharpe_std=0.3)
    print(f"Deflated Sharpe Ratio (rough, n_trials~{N_TRIALS_ROUGH_ESTIMATE} guessed, not logged): "
          f"{dsr:.1%} probability this Sharpe reflects genuine skill rather than the best "
          f"of the variations tried along the way. Below ~95% means: not yet convincing.")

    if benchmark_col:
        print("\n" + "-" * 70)
        print(f"FULL TEARSHEET -- benchmark: {benchmark_col}")
        print("-" * 70)
        sheet = compute_tearsheet(result, close_full[benchmark_col])
        print(render_tearsheet(sheet))
    else:
        print("\n(no benchmark column declared for this override file -- "
              "full tearsheet skipped; only the core metrics table above applies)")

    trade_log = derive_trade_log(result)
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trade_log.csv")
    trade_log.to_csv(log_path, index=False)
    print(f"\n{len(trade_log)} buy/sell events across {result.meta['n_rebalances']} rebalance dates "
          f"-- full log written to {os.path.normpath(log_path)}")
    if len(trade_log):
        wd_counts = trade_log["date"].dt.day_name().value_counts()
        print(f"rebalance weekdays: {dict(wd_counts)} "
              f"(non-Wednesday = a Wednesday holiday, fell back to the nearest trading day)")
        print("Last 15 trades:")
        print(trade_log.tail(15).to_string(index=False))

    # -- deliberately-broken same-bar replay, for comparison only ----------
    # Mirrors the same rules (ranking, filters, ATR sizing if available,
    # regime gate) but decides AND trades off the same day's close, with no
    # implementation lag at all -- the bug this whole comparison exists to
    # quantify.
    wednesdays = close.index[close.index.weekday == 2]
    half_spread = SPREAD_BPS / 1e4 / 2
    w = np.zeros(len(assets))
    V = 1.0
    vals = []
    for i, date in enumerate(close.index):
        if i > 0:
            r = close.iloc[i].to_numpy() / close.iloc[i - 1].to_numpy() - 1
            cash_w = 1 - w.sum()
            V *= (w * (1 + r)).sum() + cash_w
        if date in wednesdays and i >= WARMUP:
            ok = eligible.iloc[i].to_numpy() & np.isfinite(adj_slope.iloc[i].to_numpy())
            if have_atr:
                ok = ok & np.isfinite(atr_pct.iloc[i].to_numpy()) & (atr_pct.iloc[i].to_numpy() > 0)
            if ok.sum() >= MIN_NAMES:
                s = adj_slope.iloc[i].to_numpy()
                n_ok = int(ok.sum())
                k = max(1, int(round(TOP_QUANTILE * n_ok)))
                idx = np.flatnonzero(ok)
                order = np.argsort(-s[idx], kind="stable")
                chosen = idx[order[:k]]
                buys_ok = bool(regime.iloc[i]) if regime is not None else True
                if buys_ok:
                    target = np.zeros(len(assets))
                    if have_atr:
                        target[chosen] = RISK_FACTOR / atr_pct.iloc[i].to_numpy()[chosen]
                        if target.sum() > 1.0:
                            target = target / target.sum()
                    else:
                        target[chosen] = 1.0 / k
                else:
                    target = w.copy()
                    for j in np.flatnonzero(w > 1e-12):
                        if j not in set(chosen.tolist()):
                            target[j] = 0.0
                traded = np.abs(target - w).sum()
                V *= (1 - half_spread * traded)
                w = target
        vals.append(V)
    naive_nav = pd.Series(vals, index=close.index)
    naive_cagr = (naive_nav.iloc[-1] / naive_nav.iloc[0]) ** (
        365.25 / (naive_nav.index[-1] - naive_nav.index[0]).days) - 1
    engine_cagr = float(tbl.loc["engine_correct", "cagr"])

    print("\n" + "=" * 70)
    print("SAME-BAR REPLAY (deliberately broken -- decide and trade off the same close)")
    print("=" * 70)
    print(f"naive same-bar CAGR : {naive_cagr:.2%}")
    print(f"engine CAGR         : {engine_cagr:.2%}")
    print(f"leak on THIS data   : {naive_cagr - engine_cagr:+.2%} of CAGR/year")

    if benchmark_col:
        bench = close_full[benchmark_col].reindex(close.index).dropna()
        bench_cagr = (bench.iloc[-1] / bench.iloc[0]) ** (
            365.25 / (bench.index[-1] - bench.index[0]).days) - 1
        print(f"\n{benchmark_col} buy & hold CAGR over the same window: {bench_cagr:.2%} "
              f"(price-return index -- dividends excluded)")


if __name__ == "__main__":
    main()
