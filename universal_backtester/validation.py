"""Statistical rigor on top of a BacktestResult -- the piece that separates
"a backtest ran" from "this result should change anyone's mind."

A Sharpe ratio computed on one run of one parameter choice on one dataset is
close to meaningless on its own: it says nothing about how likely that
number was to arise by chance, and nothing about how much of it survives
resampling. Two checks address that:

  - `bootstrap_sharpe_ci`  -- block-bootstrap the daily returns and report a
    confidence interval on the Sharpe ratio, not just a point estimate.
  - `deflated_sharpe_ratio` -- the probability the observed Sharpe is
    genuine skill rather than the best of `n_trials` variations someone
    tried (Bailey & Lopez de Prado's deflated Sharpe ratio). If you tried
    5 parameter combinations and reported the best one, that number IS a
    trial count and belongs in this calculation -- not reporting it is how
    a backtest quietly becomes data-mined.

Neither of these makes a bad strategy good. They make it possible to tell
the difference between "this looks good" and "this is likely to still look
good out of sample," which is the only question that matters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

ANN = 252
_EULER_GAMMA = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation --
    accurate to ~1e-9, no scipy dependency needed for one function)."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


@dataclass
class BootstrapResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    n_resamples: int
    fraction_positive: float   # fraction of resamples with Sharpe > 0


def bootstrap_sharpe_ci(returns: pd.Series, block_size: int = 20, n_resamples: int = 2000,
                        confidence: float = 0.90, rf_daily: Optional[pd.Series] = None,
                        ann: int = ANN, seed: Optional[int] = None) -> BootstrapResult:
    """Block-bootstrap confidence interval on the annualized Sharpe ratio.

    Ordinary (iid) bootstrapping is wrong for return series because returns
    are autocorrelated -- momentum and mean-reversion both violate iid.
    Block bootstrap resamples contiguous chunks (`block_size` trading days)
    to preserve short-run dependence structure, which is the standard fix
    (Politis & Romano).
    """
    r = returns.dropna()
    if rf_daily is not None:
        r = r - rf_daily.reindex(r.index).fillna(0.0)
    n = len(r)
    if n < block_size * 3:
        raise ValueError(f"only {n} observations -- too short to block-bootstrap "
                         f"meaningfully at block_size={block_size}")
    rng = np.random.default_rng(seed)
    vals = r.to_numpy()
    n_blocks = int(np.ceil(n / block_size))

    def one_sharpe(sample: np.ndarray) -> float:
        sd = sample.std(ddof=1)
        return float(sample.mean() / sd * np.sqrt(ann)) if sd > 0 else float("nan")

    point = one_sharpe(vals)
    boot = np.empty(n_resamples)
    for b in range(n_resamples):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([vals[s:s + block_size] for s in starts])[:n]
        boot[b] = one_sharpe(sample)
    boot = boot[np.isfinite(boot)]

    alpha = 1 - confidence
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapResult(
        point_estimate=point, ci_low=float(lo), ci_high=float(hi),
        confidence=confidence, n_resamples=len(boot),
        fraction_positive=float((boot > 0).mean()))


def expected_max_sharpe(n_trials: int, trial_sharpe_std: float) -> float:
    """Expected value of the MAXIMUM Sharpe ratio you'd see just from trying
    `n_trials` independent, genuinely-skill-less strategies whose Sharpe
    ratios have standard deviation `trial_sharpe_std` across trials (Bailey
    & Lopez de Prado 2014, eq. 7). This is the benchmark the deflated Sharpe
    ratio compares your observed Sharpe against -- not zero.
    """
    if n_trials <= 1:
        return 0.0
    return trial_sharpe_std * (
        (1 - _EULER_GAMMA) * _norm_ppf(1 - 1.0 / n_trials)
        + _EULER_GAMMA * _norm_ppf(1 - 1.0 / (n_trials * math.e)))


def deflated_sharpe_ratio(observed_sharpe: float, n_obs: int, n_trials: int = 1,
                          trial_sharpe_std: float = 0.0, skew: float = 0.0,
                          kurtosis: float = 3.0) -> float:
    """Probability the observed (non-annualized, per-period) Sharpe ratio
    reflects genuine skill rather than the best of `n_trials` variations
    tried along the way (Bailey & Lopez de Prado's Probabilistic /
    Deflated Sharpe Ratio). Returns a probability in [0, 1] -- treat
    anything under ~0.95 as "this needs more evidence," not as a strategy
    to act on.

    `n_trials=1, trial_sharpe_std=0.0` (the defaults) collapses this to the
    ordinary Probabilistic Sharpe Ratio against a benchmark of 0 -- correct
    to use ONLY if you tried exactly one thing and are not selecting the
    best of several parameter sweeps, universes, or lookback windows. If
    you tried more than one and are reporting the best, `n_trials` MUST
    reflect that count, or this number silently overstates confidence --
    exactly the failure mode it exists to catch.

    `observed_sharpe` must be the PER-PERIOD Sharpe (e.g. daily mean/std),
    not annualized -- the sampling-distribution correction below assumes
    the same period the raw returns were measured in. Annualize the RESULT
    of your own read of the number, not the input.
    """
    if n_obs < 2:
        return float("nan")
    sr0 = expected_max_sharpe(n_trials, trial_sharpe_std) if n_trials > 1 else 0.0
    denom = math.sqrt(max(1e-12, (1 - skew * observed_sharpe
                                  + (kurtosis - 1) / 4 * observed_sharpe ** 2) / (n_obs - 1)))
    z = (observed_sharpe - sr0) / denom
    return _norm_cdf(z)


def deflated_sharpe_from_returns(returns: pd.Series, n_trials: int = 1,
                                 trial_sharpe_std: float = 0.0, ann: int = ANN,
                                 rf_daily: Optional[pd.Series] = None) -> float:
    """Convenience wrapper: compute the per-period Sharpe, skew, and excess
    kurtosis directly from a daily-return series and feed deflated_sharpe_ratio.

    Unlike `deflated_sharpe_ratio` itself (which is unit-agnostic and wants
    everything in the SAME period), `trial_sharpe_std` here is taken in
    ANNUALIZED units -- the way anyone actually thinks about "how much do
    Sharpe ratios vary across the strategies I tried" -- and converted to
    per-period internally via `/ sqrt(ann)`. Passing an annualized number
    into the raw function instead (mixing units) silently produces a wildly
    wrong answer -- observed once already while building this: the exact
    same inputs went from "0% probability of skill" to "97%" purely by
    fixing that unit mismatch. This wrapper exists so that mistake can't
    happen again through this entry point.
    """
    r = returns.dropna()
    if rf_daily is not None:
        r = r - rf_daily.reindex(r.index).fillna(0.0)
    sd = r.std(ddof=1)
    if sd == 0 or len(r) < 3:
        return float("nan")
    sr = float(r.mean() / sd)
    skew = float(r.skew())
    kurt = float(r.kurtosis()) + 3.0  # pandas reports EXCESS kurtosis; formula wants raw
    return deflated_sharpe_ratio(sr, n_obs=len(r), n_trials=n_trials,
                                 trial_sharpe_std=trial_sharpe_std / math.sqrt(ann),
                                 skew=skew, kurtosis=kurt)
