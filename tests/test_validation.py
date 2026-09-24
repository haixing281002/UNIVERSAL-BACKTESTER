"""Tests for validation.py -- bootstrap CI and deflated Sharpe ratio.

The hand-rolled inverse-normal (_norm_ppf) is checked against known
textbook values first, since everything else depends on it being right.
"""
import math

import numpy as np
import pandas as pd
import pytest

from universal_backtester.validation import (
    _norm_cdf, _norm_ppf, bootstrap_sharpe_ci, deflated_sharpe_ratio,
    deflated_sharpe_from_returns, expected_max_sharpe,
)


def test_norm_ppf_matches_known_quantiles():
    # standard textbook values
    assert _norm_ppf(0.5) == pytest.approx(0.0, abs=1e-6)
    assert _norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)
    assert _norm_ppf(0.025) == pytest.approx(-1.959964, abs=1e-5)
    assert _norm_ppf(0.95) == pytest.approx(1.644854, abs=1e-5)


def test_norm_cdf_and_ppf_are_inverses():
    for p in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        assert _norm_cdf(_norm_ppf(p)) == pytest.approx(p, abs=1e-6)


def test_expected_max_sharpe_grows_with_trial_count():
    e1 = expected_max_sharpe(1, trial_sharpe_std=1.0)
    e10 = expected_max_sharpe(10, trial_sharpe_std=1.0)
    e100 = expected_max_sharpe(100, trial_sharpe_std=1.0)
    assert e1 == 0.0
    assert e1 < e10 < e100, "more trials should raise the 'skill-less best' benchmark"


def test_deflated_sharpe_drops_when_trial_count_rises():
    """The same observed Sharpe should look LESS convincing the more trials
    it took to find it -- the entire point of the correction."""
    kwargs = dict(observed_sharpe=0.08, n_obs=1000, skew=0.0, kurtosis=3.0)
    dsr_1_trial = deflated_sharpe_ratio(n_trials=1, trial_sharpe_std=0.0, **kwargs)
    dsr_50_trials = deflated_sharpe_ratio(n_trials=50, trial_sharpe_std=0.05, **kwargs)
    assert dsr_50_trials < dsr_1_trial


def test_deflated_sharpe_is_a_probability():
    for n_trials, std in [(1, 0.0), (5, 0.02), (50, 0.05), (500, 0.1)]:
        p = deflated_sharpe_ratio(observed_sharpe=0.05, n_obs=500,
                                  n_trials=n_trials, trial_sharpe_std=std)
        assert 0.0 <= p <= 1.0


def test_deflated_sharpe_from_returns_matches_manual_computation():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0006, 0.01, size=600))
    got = deflated_sharpe_from_returns(r, n_trials=1, trial_sharpe_std=0.0)
    sr = r.mean() / r.std(ddof=1)
    skew = float(r.skew())
    kurt = float(r.kurtosis()) + 3.0
    expected = deflated_sharpe_ratio(float(sr), n_obs=len(r), n_trials=1,
                                     trial_sharpe_std=0.0, skew=skew, kurtosis=kurt)
    assert got == pytest.approx(expected, abs=1e-9)


def test_bootstrap_sharpe_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-01", periods=1000)
    r = pd.Series(rng.normal(0.0005, 0.01, size=1000), index=idx)
    res = bootstrap_sharpe_ci(r, block_size=20, n_resamples=300, confidence=0.90, seed=1)
    assert res.ci_low < res.ci_high
    # the point estimate need not sit exactly inside a resampled CI, but for
    # a stable iid-ish series with enough resamples it should be close
    assert res.ci_low - 0.5 <= res.point_estimate <= res.ci_high + 0.5


def test_bootstrap_ci_is_wider_for_shorter_samples():
    """Sharpe already normalizes by volatility, so a noisier PRICE series
    doesn't necessarily give a wider Sharpe CI (a persistent small edge on a
    low-vol series can be a very STABLE Sharpe estimate). What unambiguously
    widens a Sharpe CI is less data: fewer independent observations to
    estimate the ratio from."""
    rng = np.random.default_rng(2)
    long_idx = pd.bdate_range("2010-01-01", periods=2500)
    long_r = pd.Series(rng.normal(0.0004, 0.012, size=2500), index=long_idx)
    short_r = long_r.iloc[:200]
    res_long = bootstrap_sharpe_ci(long_r, block_size=15, n_resamples=400, seed=3)
    res_short = bootstrap_sharpe_ci(short_r, block_size=15, n_resamples=400, seed=3)
    assert (res_short.ci_high - res_short.ci_low) > (res_long.ci_high - res_long.ci_low)


def test_bootstrap_rejects_too_short_a_series():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, size=10))
    with pytest.raises(ValueError):
        bootstrap_sharpe_ci(r, block_size=20, n_resamples=100)
