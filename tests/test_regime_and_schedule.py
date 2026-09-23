"""Tests for the weekday-anchored rebalance schedule, the buys-allowed
regime gate, and the ATR risk-parity allocator -- the pieces added to
make the momentum_rotation.py example follow Clenow's actual rules
(Wednesday schedule, top-20%, ATR sizing, 200-DMA regime filter) instead
of approximating around them.
"""
import numpy as np
import pandas as pd
import pytest

from universal_backtester import Backtester, build_allocator
from universal_backtester.engine import rebalance_dates


def test_weekly_wed_picks_the_actual_wednesday_when_it_traded():
    idx = pd.bdate_range("2024-01-01", periods=60)  # Mon-Fri, no holidays
    picked = rebalance_dates(idx, "weekly_wed")
    assert (picked.weekday == 2).all(), "every picked date should be a Wednesday"


def test_weekly_wed_falls_back_to_nearest_trading_day_on_a_holiday():
    idx = pd.bdate_range("2024-01-01", periods=10)
    # Remove the first Wednesday to simulate a market holiday.
    wed = idx[idx.weekday == 2][0]
    idx2 = idx.drop(wed)
    picked = rebalance_dates(idx2, "weekly_wed")
    # That week's pick should be Tuesday (dist 1) or Thursday (dist 2) --
    # Tuesday if both are available, since it's closer.
    week_period = idx2.to_period("W")
    this_week = idx2[week_period == wed.to_period("W")]
    got = [d for d in picked if d in this_week][0]
    assert got.weekday() in (1, 3)
    if 1 in this_week.weekday:
        assert got.weekday() == 1


def test_weekly_differs_from_weekly_wed_on_the_same_data():
    """The bug this was written to fix: "weekly" alone lands on the last
    trading day of the week (usually Friday), which silently is NOT the
    same schedule as a paper's explicit "trade on Wednesday" rule."""
    idx = pd.bdate_range("2024-01-01", periods=60)
    fri_based = rebalance_dates(idx, "weekly")
    wed_based = rebalance_dates(idx, "weekly_wed")
    assert (fri_based.weekday == 4).all()
    assert (wed_based.weekday == 2).all()
    assert not fri_based.equals(wed_based)


def _regime_universe(n_names=6, n_days=400, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n_days)
    names = [f"A{i}" for i in range(n_names)]
    rets = rng.normal(0.0003, 0.012, size=(n_days, n_names))
    prices = pd.DataFrame(100.0 * np.cumprod(1.0 + rets, axis=0), index=idx, columns=names)
    eligible = pd.DataFrame(True, index=idx, columns=names)
    score = prices.pct_change(60, fill_method=None)
    return prices, eligible, names, score


def test_buys_allowed_false_never_adds_a_new_name():
    prices, eligible, names, score = _regime_universe()
    bt = Backtester(prices=prices, assets=names, spread_bps=10, lag_days=1, membership=eligible)
    alloc = build_allocator("cross_sectional", names, n_hold=3, min_names=3)
    # regime ALWAYS false after warmup: buys should never open the first position.
    regime = pd.Series(False, index=prices.index)
    res = bt.run(allocator=alloc, rebalance="monthly", alpha=score, regime=regime, warmup=65)
    assert (res.weights.abs().sum(axis=1) < 1e-9).all(), \
        "no position should ever open while buys_allowed is always False"


def test_buys_allowed_false_still_sells_a_disqualified_holding():
    prices, eligible, names, score = _regime_universe()
    # regime TRUE for a while (let it buy), then FALSE forever (block re-buys).
    cutover = prices.index[150]
    regime = pd.Series(prices.index < cutover, index=prices.index)
    bt = Backtester(prices=prices, assets=names, spread_bps=10, lag_days=1, membership=eligible)
    alloc = build_allocator("cross_sectional", names, n_hold=3, min_names=3)
    res = bt.run(allocator=alloc, rebalance="monthly", alpha=score, regime=regime, warmup=65)

    pre = res.weights.loc[:cutover].iloc[-2]
    assert (pre > 1e-9).sum() > 0, "should have opened positions while the regime was on"

    post_dates = res.weights.loc[res.weights.index > cutover]
    held_counts = (post_dates > 1e-9).sum(axis=1)
    # Held count can only ever go down (sells) or stay flat after the gate
    # closes -- it must never increase (that would mean a new buy slipped through).
    assert (held_counts.diff().dropna() <= 0).all(), \
        "held-name count increased after buys_allowed went permanently False"


def test_atr_risk_parity_leaves_cash_when_sizing_is_small():
    prices, eligible, names, score = _regime_universe()
    # Flat, tiny ATR% relative to a large risk_factor's reciprocal -> weights
    # should sum to well under 1, i.e. real cash drag, not forced full deployment.
    atr_pct = pd.DataFrame(0.30, index=prices.index, columns=names)  # 30% ATR/price -- huge, so weight per name is tiny
    bt = Backtester(prices=prices, assets=names, spread_bps=10, lag_days=1, membership=eligible)
    alloc = build_allocator("atr_risk_parity", names, n_hold=3, risk_factor=0.001, min_names=3)
    res = bt.run(allocator=alloc, rebalance="monthly", alpha=score, vols=atr_pct, warmup=65)
    on_rebal = res.weights.loc[res.rebalances]
    assert (on_rebal.sum(axis=1) < 0.05).all(), "risk-factor/ATR% sizing should leave most of NAV in cash here"


def test_atr_risk_parity_never_exceeds_full_investment():
    prices, eligible, names, score = _regime_universe()
    atr_pct = pd.DataFrame(0.001, index=prices.index, columns=names)  # tiny ATR% -> huge raw weight, must be capped
    bt = Backtester(prices=prices, assets=names, spread_bps=10, lag_days=1, membership=eligible)
    alloc = build_allocator("atr_risk_parity", names, n_hold=3, risk_factor=0.001, min_names=3)
    res = bt.run(allocator=alloc, rebalance="monthly", alpha=score, vols=atr_pct, warmup=65)
    assert (res.weights.sum(axis=1) <= 1.0 + 1e-6).all(), "long-only engine must never exceed 100% invested"
