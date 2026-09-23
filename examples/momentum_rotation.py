"""Worked example: a Clenow "Stocks on the Move"-style cross-sectional
momentum rotation, built on the generic universal_backtester engine.

Usage:
    python examples/momentum_rotation.py path/to/prices.xlsx [--sheet NAME]
    python examples/momentum_rotation.py path/to/prices.csv

The price file just needs a date column and one price column per tradable
asset (a "banner workbook" with multiple series blocks also works -- see
universal_backtester.data.load_banner_workbook). This script does not assume
any specific market, universe, or data provider.

It runs the SAME strategy two ways and prints both:
  1. correctly, through the engine (signal shifted 1+lag_days before use)
  2. a deliberately-broken same-bar replay (decide and trade off the same
     close) -- to show you, on your own data, how large that leak is.
Never trust (2); it exists only to quantify the mistake.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__file__.rsplit("/", 2)[0]))  # allow running without `pip install -e .`

from universal_backtester import Backtester, build_allocator, assert_causal, LookaheadError
from universal_backtester.data import load_wide_csv, load_banner_workbook
from universal_backtester.metrics import metrics_table, render_table
from universal_backtester.signals import rolling_regression_momentum, sma, max_abs_move_flag

REG_WINDOW = 90
SMA_WINDOW = 100
GAP_WINDOW = 90
GAP_THRESHOLD = 0.15
SPREAD_BPS = 30.0
LAG_DAYS = 1
TOP_QUANTILE = 0.30
MIN_NAMES = 3


def load(path: str, sheet: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        df, _ = load_banner_workbook(path, sheet=sheet)
    else:
        df, _ = load_wide_csv(path)
    return df.select_dtypes(include=[float, int]).dropna(how="all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("price_file")
    ap.add_argument("--sheet", default="Sheet1")
    args = ap.parse_args()

    close = load(args.price_file, args.sheet).dropna()
    assets = list(close.columns)
    print(f"Loaded {len(assets)} series, {close.index.min().date()} -> {close.index.max().date()} "
          f"({len(close)} rows)\nAssets: {assets}")

    adj_slope = pd.DataFrame(index=close.index, columns=assets, dtype=float)
    eligible = pd.DataFrame(index=close.index, columns=assets, dtype=bool)
    for name in assets:
        s, _ = rolling_regression_momentum(close[name], window=REG_WINDOW)
        adj_slope[name] = s
        above = close[name] > sma(close[name], SMA_WINDOW)
        gap = max_abs_move_flag(close[name], GAP_WINDOW, GAP_THRESHOLD)
        eligible[name] = above & (~gap)

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
    alloc = build_allocator("cross_sectional", assets, quantile=TOP_QUANTILE,
                            weighting="equal", min_names=MIN_NAMES, ascending=False)
    result = bt.run(allocator=alloc, rebalance="weekly", alpha=adj_slope,
                    name="engine_correct", warmup=REG_WINDOW)

    tbl = metrics_table([result])
    print("\n" + "=" * 70)
    print("CORRECT (causally-shifted, 30bp cost)")
    print("=" * 70)
    print(render_table(tbl))
    print(f"mean cash weight: {result.cash_weight.mean():.1%}")

    # -- deliberately-broken same-bar replay, for comparison only ----------
    wednesdays = close.index[close.index.weekday == 2]
    k = max(1, int(round(TOP_QUANTILE * len(assets))))
    half_spread = SPREAD_BPS / 1e4 / 2

    w = np.zeros(len(assets))
    V = 1.0
    vals = []
    for i, date in enumerate(close.index):
        if i > 0:
            r = close.iloc[i].to_numpy() / close.iloc[i - 1].to_numpy() - 1
            cash_w = 1 - w.sum()
            V *= (w * (1 + r)).sum() + cash_w
        if date in wednesdays and i >= REG_WINDOW:
            ok = eligible.iloc[i].to_numpy() & np.isfinite(adj_slope.iloc[i].to_numpy())
            if ok.sum() >= MIN_NAMES:
                s = adj_slope.iloc[i].to_numpy()
                order = np.argsort(-np.where(ok, s, -np.inf))
                chosen = order[:k]
                target = np.zeros(len(assets))
                target[chosen] = 1.0 / k
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


if __name__ == "__main__":
    main()
