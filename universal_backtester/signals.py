"""Reusable signal-construction helpers. These build RAW, unshifted signals --
the engine (engine.Backtester._shift_causal) is what makes them safe to trade
on; nothing here should be treated as already lag-safe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def rolling_regression_momentum(close: pd.Series, window: int = 90,
                                annualize_days: int = 250) -> tuple[pd.Series, pd.Series]:
    """Clenow-style momentum score: regress LN(price) on a day-index over
    the trailing `window` days, annualize the slope, and weight it by the
    fit (R^2) -- so a smooth, persistent trend outranks a choppy one at the
    same total return. Returns (adjusted_slope, r_squared), both raw/unshifted.
    """
    logp = np.log(close)
    n = len(logp)
    slope = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    if n >= window:
        w = sliding_window_view(logp.values, window)
        x = np.arange(window)
        xc = x - x.mean()
        Sxx = (xc ** 2).sum()
        ymean = w.mean(axis=1)
        yc = w - ymean[:, None]
        Sxy = (xc[None, :] * yc).sum(axis=1)
        b = Sxy / Sxx
        Syy = (yc ** 2).sum(axis=1)
        r2v = np.where(Syy > 0, (b ** 2 * Sxx) / Syy, 0.0)
        slope[window - 1:] = b
        r2[window - 1:] = r2v
    slope = pd.Series(slope, index=close.index)
    r2 = pd.Series(r2, index=close.index)
    adjusted = ((np.exp(slope) ** annualize_days) - 1) * r2
    return adjusted, r2


def trailing_return(close: pd.Series, window: int) -> pd.Series:
    return close.pct_change(window, fill_method=None)


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def realized_vol(close: pd.Series, window: int = 20, annualize: int = 252) -> pd.Series:
    return close.pct_change().rolling(window).std() * np.sqrt(annualize)


def max_abs_move_flag(close: pd.Series, window: int = 90, threshold: float = 0.15) -> pd.Series:
    """True where any single-day move over the trailing window exceeded
    `threshold` in absolute value -- a gap/quality disqualifier."""
    return close.pct_change().abs().rolling(window).max() > threshold


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """Average True Range: rolling mean of True Range = max(high-low,
    |high-prev_close|, |low-prev_close|). Needs genuine High/Low data --
    there is no way to approximate ATR from Close alone that isn't
    misleading, so don't call this if your price file only has Close."""
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    return tr.rolling(window).mean()


def regime_filter(benchmark_close: pd.Series, window: int = 200) -> pd.Series:
    """True on days the benchmark closes above its own trailing SMA --
    a common "risk-on" gate for blocking new entries in a downtrend."""
    return benchmark_close > sma(benchmark_close, window)
