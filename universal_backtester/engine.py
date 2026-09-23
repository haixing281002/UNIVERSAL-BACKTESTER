"""Deterministic daily portfolio-accounting engine.

Two properties it guarantees, because they are where backtests lie:

1. CAUSALITY. Every signal (alpha, eligibility, vols) is shifted by
   (1 + lag_days) before an allocator ever sees it: the weights applied to
   day t's return are decided using data knowable at t-1-lag at the latest.
   A signal is never used to trade the bar that produced it.

2. NO SURVIVORSHIP DRIFT. Either the price frame is complete for the fixed
   asset set (a NaN is treated as a data fault), or the caller passes
   `membership` to declare a changing cross-section explicitly -- and a name
   that leaves membership is SOLD at cost on the day it's learned, never
   quietly dropped and the rest renormalised for free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from universal_backtester.allocators import Allocator, AllocatorContext


class LookaheadError(RuntimeError):
    """Raised by assert_causal when a signal appears to know the future."""


def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Trading dates on which a rebalance occurs, using the LAST trading day
    of each period present in the data -- never a calendar date that may not
    actually trade."""
    if freq == "daily":
        return index
    codes = {"weekly": "W", "monthly": "M", "quarterly": "Q", "annual": "Y"}
    if freq not in codes:
        raise ValueError(f"unsupported rebalance frequency '{freq}'")
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period(codes[freq])).last().values)


def _alpha_row(alp: pd.DataFrame, date, cross_sectional: bool):
    row = alp.loc[date]
    if not cross_sectional and row.isna().any():
        return None
    return row.to_numpy(dtype=float)


@dataclass
class BacktestResult:
    name: str
    value: pd.Series
    weights: pd.DataFrame
    returns: pd.Series
    turnover: pd.Series
    costs: pd.Series
    cash_weight: pd.Series
    rebalances: pd.DatetimeIndex
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def cash_drag_days(self) -> int:
        return int((self.cash_weight > 1e-6).sum())


class Backtester:
    """Same inputs -> same bytes out, always."""

    def __init__(
        self,
        prices: pd.DataFrame,
        assets: List[str],
        rf_daily: Optional[pd.Series] = None,
        spread_bps: float = 10.0,
        lag_days: int = 0,
        allow_cash: bool = True,
        ann: int = 252,
        membership: Optional[pd.DataFrame] = None,
    ):
        missing = [a for a in assets if a not in prices.columns]
        if missing:
            raise KeyError(f"prices frame is missing assets: {missing}")
        self.assets = list(assets)
        self.prices = prices[self.assets].copy()

        self.membership = None
        if membership is not None:
            miss = [a for a in self.assets if a not in membership.columns]
            if miss:
                raise KeyError(f"membership frame is missing assets: {miss}")
            self.membership = (membership[self.assets]
                               .reindex(self.prices.index).fillna(False).astype(bool))
        elif self.prices.isna().any().any():
            raise ValueError(
                "price frame contains NaNs for the fixed asset set. Either fill/trim "
                "the data so the investable universe is complete, or pass "
                "`membership=` (a boolean frame saying who was investable when) if "
                "this IS a changing cross-section -- names listing and delisting.")
        self.returns = self.prices.pct_change(fill_method=None)
        self.rf = (rf_daily.reindex(self.prices.index).fillna(0.0)
                   if rf_daily is not None else pd.Series(0.0, index=self.prices.index))
        self.half_spread = float(spread_bps) / 1e4 / 2.0
        self.lag_days = int(lag_days)
        self.allow_cash = bool(allow_cash)
        self.ann = ann
        self.n = len(self.assets)

    def _shift_causal(self, obj, extra_lag: int = 0):
        """Shift so a signal is knowable strictly before the bar it trades.
        Total shift = 1 + lag_days: the +1 removes same-bar use of the close
        that produced the signal; lag_days is any extra implementation lag."""
        return obj.shift(1 + self.lag_days + extra_lag)

    def run(
        self,
        allocator: Allocator,
        rebalance: str = "monthly",
        alpha: Optional[pd.DataFrame] = None,
        vols: Optional[pd.DataFrame] = None,
        rf_horizon_days: int = 21,
        name: str = "strategy",
        warmup: int = 0,
    ) -> BacktestResult:
        idx = self.prices.index
        rebal = set(rebalance_dates(idx, rebalance))

        alp = self._shift_causal(alpha)[self.assets] if alpha is not None else None
        vol_shift = self._shift_causal(vols[self.assets]) if vols is not None else None

        elig_arr = None
        if self.membership is not None:
            elig = self._shift_causal(self.membership).fillna(False).astype(bool)
            elig_arr = (elig & self.prices.notna()).to_numpy(dtype=bool)

        rf_period = (1.0 + self.rf).rolling(rf_horizon_days).apply(np.prod, raw=True) - 1.0
        rf_period = rf_period.fillna(0.0)

        V = 1.0
        w = np.zeros(self.n)
        first_rebal_done = False
        vals, wts, tos, cst, csh = [], [], [], [], []
        used_rebals: List[pd.Timestamp] = []
        stale = 0
        n_forced = 0
        rets_arr = self.returns.to_numpy(dtype=float)

        for i, date in enumerate(idx):
            if i == 0 or not first_rebal_done:
                gross_ret = 0.0
            else:
                r = np.nan_to_num(rets_arr[i], nan=0.0)
                cash_w = 1.0 - w.sum()
                gross = float(np.dot(w, 1.0 + r) + cash_w * (1.0 + self.rf.iloc[i]))
                new_V = V * gross
                w = (V * w * (1.0 + r)) / new_V
                gross_ret = new_V / V - 1.0
                V = new_V
                if elig_arr is not None:
                    stale += int(np.count_nonzero((w > 0) & np.isnan(rets_arr[i])))

            day_to = 0.0
            day_cost = 0.0

            if elig_arr is not None and first_rebal_done:
                dead = (w > 1e-12) & ~elig_arr[i]
                if dead.any():
                    forced = float(w[dead].sum())
                    day_cost += self.half_spread * forced
                    day_to += 0.5 * forced
                    V *= (1.0 - self.half_spread * forced)
                    w = np.where(dead, 0.0, w)
                    n_forced += int(dead.sum())

            if date in rebal and i >= warmup:
                ctx = AllocatorContext(
                    date=date,
                    current_weights=w.copy(),
                    alpha=(_alpha_row(alp, date, elig_arr is not None)
                           if alp is not None else None),
                    eligible=(elig_arr[i] if elig_arr is not None else None),
                    vols=(vol_shift.iloc[i].to_numpy(dtype=float)
                          if vol_shift is not None else None),
                    rf_period=float(rf_period.iloc[i]),
                    extras={},
                )
                if allocator.ready(ctx):
                    target = np.asarray(allocator.target_weights(ctx), dtype=float)
                    if target.shape != (self.n,):
                        raise ValueError(
                            f"{allocator.template_name} returned {target.shape}, expected {(self.n,)}")
                    target = np.clip(target, 0.0, None)
                    if not self.allow_cash:
                        tot = target.sum()
                        target = target / tot if tot > 0 else np.ones(self.n) / self.n
                    if target.sum() > 1.0 + 1e-9:
                        target = target / target.sum()

                    traded = np.abs(target - w).sum()
                    day_cost = self.half_spread * traded
                    day_to = 0.5 * traded
                    V = V * (1.0 - day_cost)
                    w = target
                    first_rebal_done = True
                    used_rebals.append(date)

            vals.append(V)
            wts.append(w.copy())
            tos.append(day_to)
            cst.append(day_cost)
            csh.append(1.0 - w.sum())

        value = pd.Series(vals, index=idx, name=name)
        weights = pd.DataFrame(wts, index=idx, columns=self.assets)
        rets = value.pct_change().fillna(0.0)

        meta = {"template": allocator.template_name, "params": allocator.params,
                "rebalance": rebalance, "lag_days": self.lag_days,
                "spread_bps": self.half_spread * 2 * 1e4, "allow_cash": self.allow_cash,
                "n_rebalances": len(used_rebals), "warmup_days": warmup}
        if elig_arr is not None:
            meta["changing_universe"] = True
            meta["forced_exits"] = n_forced
            meta["stale_marks"] = stale
            meta["mean_eligible"] = float(elig_arr.sum(axis=1).mean())
        if hasattr(allocator, "diagnostics"):
            meta["allocator_diagnostics"] = allocator.diagnostics()

        return BacktestResult(
            name=name, value=value, weights=weights, returns=rets,
            turnover=pd.Series(tos, index=idx), costs=pd.Series(cst, index=idx),
            cash_weight=pd.Series(csh, index=idx),
            rebalances=pd.DatetimeIndex(used_rebals), meta=meta)


def assert_causal(signal, returns, tol: float = 0.35, label: str = "signal",
                  horizons: Sequence[int] = (0, 1)) -> None:
    """Look-ahead tripwire: correlate signal[t] with returns[t+h].

      h = 0  the signal knows the bar it trades (an off-by-one in the shift)
      h = 1  the signal knows tomorrow (a negative shift left in by accident)

    This does not PROVE causality -- only the engine's shift logic does --
    but it catches the mistakes that silently inflate every downstream
    number, cheaply enough to run on every input to every backtest.
    """
    s = pd.DataFrame(signal).astype(float)
    r = pd.DataFrame(returns).astype(float)
    for h in horizons:
        rh = r.shift(-h)
        common = s.index.intersection(rh.index)
        for c in s.columns:
            if c not in rh.columns:
                continue
            a, b = s.loc[common, c], rh.loc[common, c]
            m = a.notna() & b.notna()
            if m.sum() < 60:
                continue
            sa, sb = a[m].std(), b[m].std()
            if sa == 0 or sb == 0:
                continue
            rho = float(np.corrcoef(a[m], b[m])[0, 1])
            if abs(rho) > tol:
                raise LookaheadError(
                    f"{label}/{c}: |corr(signal[t], return[t+{h}])| = {abs(rho):.2f} > {tol}. "
                    f"The signal appears to know {'the bar it trades' if h == 0 else f'{h} bar(s) ahead'}.")
