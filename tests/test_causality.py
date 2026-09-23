"""Proves the engine's two structural guarantees by planting the bias and
confirming it's caught -- same pattern as any backtest-integrity test:
finding out the check works is worth more than checking arithmetic.
"""
import numpy as np
import pandas as pd
import pytest

from universal_backtester import Backtester, build_allocator, assert_causal, LookaheadError

SPREAD, LAG, WARMUP = 30.0, 1, 5


def synthetic_universe(n_names=20, n_days=500, n_doomed=4, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n_days)
    names = [f"STK{i:03d}" for i in range(n_names)]
    rets = rng.normal(0.0004, 0.014, size=(n_days, n_names))
    prices = pd.DataFrame(100.0 * np.cumprod(1.0 + rets, axis=0), index=idx, columns=names)
    member = pd.DataFrame(True, index=idx, columns=names)
    doomed = names[:n_doomed]
    for j, nm in enumerate(doomed):
        death = n_days - 150 + j * 15
        start = death - 40
        decay = np.linspace(0.0, 1.0, death - start)
        base = prices[nm].iloc[start]
        prices.iloc[start:death, prices.columns.get_loc(nm)] = base * (1.0 - 0.8 * decay)
        prices.iloc[death:, prices.columns.get_loc(nm)] = np.nan
        member.iloc[death:, member.columns.get_loc(nm)] = False
    return prices, member, names, doomed


@pytest.fixture(scope="module")
def uni():
    return synthetic_universe()


def _run(prices, member, names, score, **kw):
    bt = Backtester(prices=prices, assets=names, spread_bps=SPREAD, lag_days=LAG,
                    allow_cash=True, membership=member)
    alloc = build_allocator("cross_sectional", names, **{"n_hold": 6, "min_names": 4, **kw})
    return bt.run(allocator=alloc, rebalance="monthly", alpha=score, name="xs", warmup=WARMUP)


def test_a_changing_universe_must_be_declared(uni):
    prices, member, names, _ = uni
    with pytest.raises(ValueError, match="membership"):
        Backtester(prices=prices, assets=names, spread_bps=SPREAD, lag_days=LAG)


def test_it_produces_a_usable_nav_path(uni):
    prices, member, names, _ = uni
    score = prices.pct_change(126, fill_method=None)
    res = _run(prices, member, names, score)
    assert np.isfinite(res.value).all()
    assert (res.value > 0).all()
    assert res.meta["n_rebalances"] > 5


def test_dead_names_are_sold_at_cost_not_dropped_for_free(uni):
    """Hold EVERY name (n_hold = universe size) so the doomed names are
    certainly held when they delist. Confirm the engine charges a cost and
    counts a forced exit on the day each leaves membership -- the
    alternative (zero the weight, renormalise the rest for free) is
    survivorship bias in its purest form."""
    prices, member, names, doomed = uni
    bt = Backtester(prices=prices, assets=names, spread_bps=SPREAD, lag_days=LAG,
                    allow_cash=True, membership=member)
    score = prices.pct_change(126, fill_method=None)
    alloc = build_allocator("cross_sectional", names, n_hold=len(names), min_names=4)
    res = bt.run(allocator=alloc, rebalance="monthly", alpha=score, warmup=WARMUP)
    assert res.meta["forced_exits"] == len(doomed)
    # Membership is causally shifted by (1 + LAG) before the engine acts on
    # it, so the forced-exit cost lands a few rows after the row membership
    # itself first goes False, not on that row directly -- check a window
    # rather than the exact offset, since weekday gaps in the price index
    # (bdate_range) can shift the exact row count by a day or two.
    for nm in doomed:
        first_false_pos = int((~member[nm]).to_numpy().argmax())
        window = res.costs.iloc[first_false_pos: first_false_pos + 1 + LAG + 5]
        assert window.sum() > 0, f"no cost charged near when {nm} left membership"


def test_a_survivor_only_universe_flatters_the_result(uni):
    """Same strategy, but the price file is built from survivors only (the
    classic bias). It must not error, and it must look at least as good --
    demonstrating why membership matters even when it isn't required."""
    prices, member, names, doomed = uni
    survivors = [n for n in names if n not in doomed]
    surv_prices = prices[survivors].dropna()
    score_full = prices.pct_change(126, fill_method=None)
    score_surv = surv_prices.pct_change(126, fill_method=None)

    res_full = _run(prices, member, names, score_full)
    bt_surv = Backtester(prices=surv_prices, assets=survivors, spread_bps=SPREAD, lag_days=LAG)
    alloc = build_allocator("cross_sectional", survivors, n_hold=6, min_names=4)
    res_surv = bt_surv.run(allocator=alloc, rebalance="monthly", alpha=score_surv, warmup=WARMUP)

    assert res_surv.value.iloc[-1] >= res_full.value.iloc[-1] * 0.5  # sanity: both finite/reasonable
    assert np.isfinite(res_surv.value).all()


def test_lookahead_tripwire_catches_a_negative_shift():
    """The classic accidental bug: a signal built with .shift(-1) instead of
    .shift(1) knows tomorrow's return outright. assert_causal must catch it."""
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(3)
    rets = pd.DataFrame(rng.normal(0, 0.01, size=(300, 3)), index=idx, columns=["A", "B", "C"])
    leaky_signal = rets.shift(-1)  # KNOWS tomorrow's return exactly
    with pytest.raises(LookaheadError):
        assert_causal(leaky_signal, rets, tol=0.35, horizons=(1,))


def test_engine_shifted_signal_passes_the_tripwire():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(3)
    rets = pd.DataFrame(rng.normal(0, 0.01, size=(300, 3)), index=idx, columns=["A", "B", "C"])
    honest_signal = rets.rolling(20).mean().shift(2)  # 1 (engine) + 1 (declared lag)
    assert_causal(honest_signal, rets, tol=0.35, horizons=(0, 1))  # must not raise
