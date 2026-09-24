"""Extended risk/return tearsheet -- the standard ratio set a factsheet
shows beyond metrics.py's core CAGR/vol/Sharpe/drawdown.

Every benchmark-relative number here (alpha, tracking error, capture
ratios, conditional returns) needs a benchmark return series. This repo's
NIFTY 500 series is PRICE-RETURN (dividends excluded), not the Total
Return Index. TRI is the correct, industry-standard comparison for a
long-only equity strategy -- a price-return index structurally understates
the benchmark by its dividend yield, roughly 1.3-1.5%/year for Indian
large/mid caps. Every number below computed against "the benchmark" is
computed against this price-return proxy because that is the only NIFTY
500 series this repo holds, and is systematically flattering to the
strategy relative to a true TRI comparison by about that much. It is fine
for comparing two strategies against the same (biased) benchmark; it is
not fine for a standalone claim like "beat the index by X%" without that
caveat attached.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from universal_backtester.metrics import cagr, ann_vol, drawdown_series, max_drawdown, sharpe as _sharpe

ANN = 252
MONTHLY = 12


def _rf_cagr(rf_daily: Optional[pd.Series], index: pd.DatetimeIndex, ann: int = ANN) -> float:
    if rf_daily is None:
        return 0.0
    rf = rf_daily.reindex(index).fillna(0.0)
    if len(rf) < 2:
        return 0.0
    years = (index[-1] - index[0]).days / 365.25
    if years <= 0:
        return 0.0
    total = float(np.prod(1.0 + rf.to_numpy(dtype=float)))
    return total ** (1.0 / years) - 1.0


TRI_CAVEAT = (
    "Benchmark is NIFTY 500 PRICE-RETURN (dividends excluded), used as a "
    "proxy for NIFTY 500 TRI because that is the only series this repo "
    "holds. Understates the true benchmark by ~1.3-1.5%/year -- every "
    "benchmark-relative number below inherits that bias in the strategy's favor."
)


def _monthly_returns(returns: pd.Series) -> pd.Series:
    r = returns.dropna()
    return (1.0 + r).resample("ME").prod() - 1.0


def positive_volatility(returns: pd.Series, ann: int = MONTHLY) -> float:
    """Std dev of positive MONTHLY returns, annualized -- "upside" volatility."""
    m = _monthly_returns(returns)
    pos = m[m > 0]
    return float(pos.std(ddof=1) * np.sqrt(ann)) if len(pos) > 2 else float("nan")


def negative_volatility(returns: pd.Series, ann: int = MONTHLY) -> float:
    m = _monthly_returns(returns)
    neg = m[m < 0]
    return float(neg.std(ddof=1) * np.sqrt(ann)) if len(neg) > 2 else float("nan")


def average_annual_max_drawdown(value: pd.Series) -> float:
    """Mean of each calendar year's worst point on the continuous
    (whole-history) drawdown curve -- not a per-year reset."""
    dd = drawdown_series(value)
    yearly_min = dd.groupby(dd.index.year).min()
    return float(-yearly_min.mean()) if len(yearly_min) else float("nan")


def beta(returns: pd.Series, bench_returns: pd.Series) -> float:
    r, b = returns.align(bench_returns, join="inner")
    m = r.notna() & b.notna()
    r, b = r[m], b[m]
    if len(r) < 3 or b.var(ddof=1) == 0:
        return float("nan")
    return float(np.cov(r, b, ddof=1)[0, 1] / b.var(ddof=1))


def adjusted_beta(returns: pd.Series, bench_returns: pd.Series) -> float:
    """Bloomberg-style shrinkage toward 1: (2/3)*raw_beta + (1/3)*1."""
    raw = beta(returns, bench_returns)
    return float((2.0 / 3.0) * raw + (1.0 / 3.0)) if np.isfinite(raw) else float("nan")


def treynor_ratio(value: pd.Series, returns: pd.Series, bench_returns: pd.Series,
                  rf_daily: Optional[pd.Series] = None, ann: int = ANN) -> float:
    b = beta(returns, bench_returns)
    if not np.isfinite(b) or b == 0:
        return float("nan")
    rf_cagr = _rf_cagr(rf_daily, value.index, ann) if rf_daily is not None else 0.0
    return float((cagr(value, ann) - rf_cagr) / b)


def alpha_adj_beta(value: pd.Series, returns: pd.Series, bench_value: pd.Series,
                   bench_returns: pd.Series, rf_daily: Optional[pd.Series] = None,
                   ann: int = ANN) -> float:
    """Jensen's alpha using the Bloomberg-adjusted beta:
    alpha = portfolio_CAGR - [rf + adj_beta * (benchmark_CAGR - rf)]."""
    b = adjusted_beta(returns, bench_returns)
    if not np.isfinite(b):
        return float("nan")
    rf_cagr = _rf_cagr(rf_daily, value.index, ann) if rf_daily is not None else 0.0
    bench_cagr = cagr(bench_value, ann)
    return float(cagr(value, ann) - (rf_cagr + b * (bench_cagr - rf_cagr)))


def tracking_error(returns: pd.Series, bench_returns: pd.Series, ann: int = ANN) -> float:
    r, b = returns.align(bench_returns, join="inner")
    diff = (r - b).dropna()
    return float(diff.std(ddof=1) * np.sqrt(ann)) if len(diff) > 2 else float("nan")


def downside_deviation(returns: pd.Series, mar: float = 0.0, ann: int = ANN) -> float:
    r = returns.dropna()
    diff = np.minimum(r.to_numpy() - mar / ann, 0.0)
    return float(np.sqrt((diff ** 2).mean()) * np.sqrt(ann))


def sortino_ratio(returns: pd.Series, mar: float = 0.0, ann: int = ANN,
                  rf_daily: Optional[pd.Series] = None) -> float:
    r = returns.dropna()
    excess = r - (rf_daily.reindex(r.index).fillna(0.0) if rf_daily is not None else 0.0)
    dd = downside_deviation(excess, mar=mar, ann=ann)
    if dd <= 0:
        return float("nan")
    return float((excess.mean() * ann - mar) / dd)


def calmar_ratio(value: pd.Series, ann: int = ANN) -> float:
    mdd = max_drawdown(value)
    return float(cagr(value, ann) / mdd) if mdd > 0 else float("nan")


def sterling_ratio(value: pd.Series, ann: int = ANN) -> float:
    """CAGR / average annual max drawdown (the simple, no-10%-adjustment
    convention -- some vendors subtract a further 10 points from the
    denominator; this does not, documented here rather than silently picked)."""
    aamdd = average_annual_max_drawdown(value)
    return float(cagr(value, ann) / aamdd) if aamdd > 0 else float("nan")


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    r = returns.dropna()
    excess = r - threshold
    gains = float(excess[excess > 0].sum())
    losses = float(-excess[excess < 0].sum())
    return float(gains / losses) if losses > 0 else float("nan")


def gain_to_pain_ratio(returns: pd.Series) -> float:
    """Schwager's definition: sum of ALL monthly returns / abs(sum of
    losing months' returns)."""
    m = _monthly_returns(returns)
    losses = float(-m[m < 0].sum())
    return float(m.sum() / losses) if losses > 0 else float("nan")


def tail_ratio(returns: pd.Series, pct: float = 0.05) -> float:
    r = returns.dropna()
    right = np.percentile(r, 100 * (1 - pct))
    left = np.percentile(r, 100 * pct)
    return float(abs(right) / abs(left)) if left != 0 else float("nan")


def conditional_return(returns: pd.Series, bench_returns: pd.Series,
                       positive: bool, ann: int = ANN) -> float:
    """Mean daily portfolio return on days the benchmark was up (or down),
    reported as an annualized run-rate (mean * ann) for comparability --
    not a compounded total, since these days aren't contiguous."""
    r, b = returns.align(bench_returns, join="inner")
    mask = (b > 0) if positive else (b < 0)
    sub = r[mask]
    return float(sub.mean() * ann) if len(sub) else float("nan")


def _capture(monthly_r: pd.Series, monthly_b: pd.Series, mask: pd.Series) -> float:
    port = float((1.0 + monthly_r[mask]).prod() - 1.0)
    bench = float((1.0 + monthly_b[mask]).prod() - 1.0)
    return float(port / bench * 100.0) if bench != 0 else float("nan")


def _monthly_pair(returns: pd.Series, bench_returns: pd.Series):
    """Capture ratios compound MONTHLY returns, not daily. Compounding a
    non-contiguous subset of DAILY returns (e.g. "every day the benchmark
    was up") explodes: cherry-picking ~half the days of a multi-year daily
    series and compounding them back-to-back as if they were consecutive
    produces a wildly nonlinear number with no real-world meaning. Monthly
    granularity is the standard convention (Morningstar-style Up/Down
    Capture) precisely because it keeps this distortion small."""
    r, b = returns.align(bench_returns, join="inner")
    mr = (1.0 + r).resample("ME").prod() - 1.0
    mb = (1.0 + b).resample("ME").prod() - 1.0
    return mr, mb


def upside_capture(returns: pd.Series, bench_returns: pd.Series) -> float:
    mr, mb = _monthly_pair(returns, bench_returns)
    return _capture(mr, mb, mb > 0)


def downside_capture(returns: pd.Series, bench_returns: pd.Series) -> float:
    mr, mb = _monthly_pair(returns, bench_returns)
    return _capture(mr, mb, mb < 0)


def capture_ratio(returns: pd.Series, bench_returns: pd.Series) -> float:
    up, down = upside_capture(returns, bench_returns), downside_capture(returns, bench_returns)
    return float(up / down) if np.isfinite(down) and down != 0 else float("nan")


def extreme_capture_ratio(returns: pd.Series, bench_returns: pd.Series, pct: float = 0.10) -> float:
    """Capture ratio restricted to the most extreme benchmark MONTHS (top/
    bottom `pct` by benchmark monthly return), rather than every up/down month."""
    mr, mb = _monthly_pair(returns, bench_returns)
    hi, lo = mb.quantile(1 - pct), mb.quantile(pct)
    up = _capture(mr, mb, mb >= hi)
    down = _capture(mr, mb, mb <= lo)
    return float(up / down) if np.isfinite(down) and down != 0 else float("nan")


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    r = returns.dropna()
    return float(-np.percentile(r, 100 * (1 - confidence)))


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    r = returns.dropna()
    cutoff = np.percentile(r, 100 * (1 - confidence))
    tail = r[r <= cutoff]
    return float(-tail.mean()) if len(tail) else float("nan")


@dataclass
class Tearsheet:
    annualized_volatility: float
    positive_volatility: float
    negative_volatility: float
    max_drawdown: float
    average_annual_max_drawdown: float
    sharpe_ratio: float
    treynor_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    sterling_ratio: float
    omega_ratio: float
    gain_to_pain_ratio: float
    tail_ratio: float
    alpha_adj_beta: float
    tracking_error: float
    ret_positive_benchmark_days: float
    ret_negative_benchmark_days: float
    upside_capture: float
    downside_capture: float
    capture_ratio: float
    extreme_capture_ratio: float
    skewness: float
    kurtosis: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_LABELS = {
    "annualized_volatility": "Annualized Volatility",
    "positive_volatility": "Positive Volatility",
    "negative_volatility": "Negative Volatility",
    "max_drawdown": "Max Drawdown",
    "average_annual_max_drawdown": "Average Annual Max Drawdown",
    "sharpe_ratio": "Sharpe Ratio",
    "treynor_ratio": "Treynor Ratio",
    "sortino_ratio": "Sortino Ratio",
    "calmar_ratio": "Calmar Ratio",
    "sterling_ratio": "Sterling Ratio",
    "omega_ratio": "Omega Ratio",
    "gain_to_pain_ratio": "Gain to Pain Ratio",
    "tail_ratio": "Tail Ratio",
    "alpha_adj_beta": "Alpha (Adj Beta)",
    "tracking_error": "Tracking Error",
    "ret_positive_benchmark_days": "Ret +Ve Nifty 500 (PR proxy for TRI)",
    "ret_negative_benchmark_days": "Ret -Ve Nifty 500 (PR proxy for TRI)",
    "upside_capture": "Upside Capture",
    "downside_capture": "Downside Capture",
    "capture_ratio": "Capture Ratio",
    "extreme_capture_ratio": "Extreme Capture Ratio",
    "skewness": "Skewness",
    "kurtosis": "Kurtosis",
    "var_95": "95% VaR",
    "var_99": "99% VaR",
    "cvar_95": "95% CVaR",
    "cvar_99": "99% CVaR",
}

_PCT_FIELDS = {"annualized_volatility", "positive_volatility", "negative_volatility",
              "max_drawdown", "average_annual_max_drawdown", "tracking_error",
              "ret_positive_benchmark_days", "ret_negative_benchmark_days",
              "var_95", "var_99", "cvar_95", "cvar_99"}
_RATIO_FIELDS = {"upside_capture", "downside_capture", "capture_ratio", "extreme_capture_ratio"}


def compute_tearsheet(result, bench_close: pd.Series, rf_daily: Optional[pd.Series] = None,
                      ann: int = ANN) -> Tearsheet:
    """`result` is a BacktestResult (or anything with .value and .returns).
    `bench_close` is the benchmark's raw price series -- returns and an
    aligned value path are derived from it here."""
    r = result.returns
    bench_close = bench_close.reindex(result.value.index)
    bench_returns = bench_close.pct_change()
    bench_value = bench_close / bench_close.dropna().iloc[0]

    return Tearsheet(
        annualized_volatility=ann_vol(r, ann),
        positive_volatility=positive_volatility(r),
        negative_volatility=negative_volatility(r),
        max_drawdown=max_drawdown(result.value),
        average_annual_max_drawdown=average_annual_max_drawdown(result.value),
        sharpe_ratio=_sharpe(result.value, r, rf_daily, ann),
        treynor_ratio=treynor_ratio(result.value, r, bench_returns, rf_daily, ann),
        sortino_ratio=sortino_ratio(r, ann=ann, rf_daily=rf_daily),
        calmar_ratio=calmar_ratio(result.value, ann),
        sterling_ratio=sterling_ratio(result.value, ann),
        omega_ratio=omega_ratio(r),
        gain_to_pain_ratio=gain_to_pain_ratio(r),
        tail_ratio=tail_ratio(r),
        alpha_adj_beta=alpha_adj_beta(result.value, r, bench_value, bench_returns, rf_daily, ann),
        tracking_error=tracking_error(r, bench_returns, ann),
        ret_positive_benchmark_days=conditional_return(r, bench_returns, positive=True, ann=ann),
        ret_negative_benchmark_days=conditional_return(r, bench_returns, positive=False, ann=ann),
        upside_capture=upside_capture(r, bench_returns),
        downside_capture=downside_capture(r, bench_returns),
        capture_ratio=capture_ratio(r, bench_returns),
        extreme_capture_ratio=extreme_capture_ratio(r, bench_returns),
        skewness=float(r.dropna().skew()),
        kurtosis=float(r.dropna().kurtosis()),
        var_95=historical_var(r, 0.95),
        var_99=historical_var(r, 0.99),
        cvar_95=historical_cvar(r, 0.95),
        cvar_99=historical_cvar(r, 0.99),
    )


def render_tearsheet(ts: Tearsheet) -> str:
    lines = []
    for field, label in _LABELS.items():
        v = getattr(ts, field)
        if not np.isfinite(v):
            s = "--"
        elif field in _PCT_FIELDS:
            s = f"{v:.2%}"
        elif field in _RATIO_FIELDS:
            s = f"{v:.1f}"
        else:
            s = f"{v:.2f}"
        lines.append(f"  {label:<38s} {s:>10s}")
    lines.append(f"\n  {TRI_CAVEAT}")
    return "\n".join(lines)
