"""Allocator templates: turn a signal into target portfolio weights.

Long-only by construction (weights are clipped at zero). Add a new strategy
shape by adding an Allocator subclass here -- the engine, the causal shifting
and the cost accounting never need to change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_REGISTRY: Dict[str, type] = {}


def register(name: str):
    def deco(cls):
        cls.template_name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def build_allocator(name: str, assets: List[str], **params) -> "Allocator":
    if name not in _REGISTRY:
        raise KeyError(f"unknown allocator '{name}'. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](assets, **params)


def list_allocators() -> List[str]:
    return sorted(_REGISTRY)


@dataclass
class AllocatorContext:
    """Everything an allocator may look at on a rebalance date. Every array
    here has already been shifted so it is knowable strictly before the bar
    it is used to trade -- see engine.Backtester._shift_causal."""
    date: pd.Timestamp
    current_weights: np.ndarray
    alpha: Optional[np.ndarray] = None       # ranking / expected-return score
    eligible: Optional[np.ndarray] = None    # bool: investable, as known on this date
    vols: Optional[np.ndarray] = None        # per-asset trailing vol (or ATR%, see allocator)
    buys_allowed: bool = True                # False = a "regime filter" is blocking NEW entries
    rf_period: float = 0.0
    extras: Optional[Dict[str, Any]] = None


class Allocator:
    template_name = "abstract"
    requires: tuple = ()   # AllocatorContext fields that must be non-None to trade

    def __init__(self, assets: List[str], **params):
        self.assets = list(assets)
        self.n = len(assets)
        self.params = params

    def ready(self, ctx: AllocatorContext) -> bool:
        return all(getattr(ctx, f) is not None for f in self.requires)

    def target_weights(self, ctx: AllocatorContext) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"template": self.template_name, "assets": self.assets, "params": self.params}


def _sell_only(current_weights: np.ndarray, chosen_idx: np.ndarray) -> np.ndarray:
    """When new buys are blocked (a regime filter is negative): zero out any
    currently-held name that fell out of the qualifying set (`chosen_idx`)
    -- a real, name-specific sell trigger still fires -- but never add a
    name that wasn't already held, and never resize a survivor. The cash
    freed by a sell is simply left in cash, not redistributed. This is the
    common paper pattern "block new entries in a downtrend, don't force an
    exit because of it."
    """
    target = current_weights.copy()
    chosen_set = set(int(i) for i in chosen_idx)
    held_idx = np.flatnonzero(target > 1e-12)
    for i in held_idx:
        if int(i) not in chosen_set:
            target[i] = 0.0
    return target


@register("equal_weight")
class EqualWeight(Allocator):
    """Equal weight across all eligible assets. Simplest possible baseline."""
    requires = ("eligible",)

    def target_weights(self, ctx: AllocatorContext) -> np.ndarray:
        ok = np.asarray(ctx.eligible, dtype=bool)
        w = np.zeros(self.n)
        if ok.sum() > 0:
            w[ok] = 1.0 / ok.sum()
        return w


@register("cross_sectional")
class CrossSectional(Allocator):
    """Rank a cross-section by a score, hold the top slice, weight it.

    Long-only: score every asset, sort, buy the winners. This is the shape
    almost every equity-momentum / factor paper takes.

    Parameters
    ----------
    n_hold      : how many names to hold. Mutually exclusive with `quantile`.
    quantile    : top fraction to hold (0.2 = top quintile of ELIGIBLE names).
    weighting   : "equal" | "signal" | "inverse_vol"
    max_weight  : per-name cap, applied after weighting and renormalised.
    ascending   : True if a LOW score is good (e.g. cheapness, low vol).
    min_names   : refuse to trade below this many eligible names (holds
                  current weights instead) -- ranking 3 names into a top
                  decile produces a number, not a portfolio.
    """
    requires = ("alpha", "eligible")

    def __init__(self, assets, n_hold=None, quantile=None, weighting="equal",
                 max_weight=None, ascending=False, min_names=5, **kw):
        super().__init__(assets, n_hold=n_hold, quantile=quantile,
                         weighting=weighting, max_weight=max_weight,
                         ascending=ascending, min_names=min_names, **kw)
        if (n_hold is None) == (quantile is None):
            raise ValueError("cross_sectional needs exactly one of n_hold or quantile")
        if weighting not in ("equal", "signal", "inverse_vol"):
            raise ValueError(f"unknown weighting '{weighting}'")
        self.n_hold = int(n_hold) if n_hold is not None else None
        self.quantile = float(quantile) if quantile is not None else None
        self.weighting = weighting
        self.max_weight = float(max_weight) if max_weight is not None else None
        self.ascending = bool(ascending)
        self.min_names = int(min_names)
        self._skipped = 0
        self._held: List[int] = []

    def target_weights(self, ctx: AllocatorContext) -> np.ndarray:
        score = np.asarray(ctx.alpha, dtype=float)
        ok = np.asarray(ctx.eligible, dtype=bool) & np.isfinite(score)
        n_ok = int(ok.sum())

        if n_ok < self.min_names:
            self._skipped += 1
            return ctx.current_weights.copy()

        k = self.n_hold if self.n_hold is not None else max(1, int(round(self.quantile * n_ok)))
        k = min(k, n_ok)

        idx = np.flatnonzero(ok)
        s = score[idx]
        order = np.argsort(s if self.ascending else -s, kind="stable")
        chosen = idx[order[:k]]
        self._held.append(k)

        if not ctx.buys_allowed:
            return _sell_only(ctx.current_weights, chosen)

        w = np.zeros(self.n)
        if self.weighting == "equal":
            w[chosen] = 1.0 / k
        elif self.weighting == "signal":
            v = score[chosen]
            v = (v.max() - v) if self.ascending else (v - v.min())
            w[chosen] = (v / v.sum()) if v.sum() > 0 else 1.0 / k
        else:  # inverse_vol
            if ctx.vols is None:
                w[chosen] = 1.0 / k
            else:
                sd = np.asarray(ctx.vols, dtype=float)[chosen]
                inv = np.where(np.isfinite(sd) & (sd > 0), 1.0 / sd, np.nan)
                w[chosen] = (np.nan_to_num(inv) / np.nansum(inv)
                             if np.isfinite(inv).any() else 1.0 / k)

        if self.max_weight is not None:
            w = np.minimum(w, self.max_weight)
            tot = w.sum()
            if tot > 0:
                w = w / tot
        return w

    def diagnostics(self) -> Dict[str, Any]:
        return {"rebalances_skipped_thin_universe": self._skipped,
                "mean_names_held": (float(np.mean(self._held)) if self._held else 0.0)}


@register("time_series_momentum")
class TimeSeriesMomentum(Allocator):
    """Long-only absolute momentum: hold assets with positive trailing
    return, sized inverse to their own volatility. No cross-section needed --
    each asset is judged only against its own history.
    """
    requires = ("alpha", "vols")

    def __init__(self, assets, min_positive=1, vol_weighted=True, **kw):
        super().__init__(assets, min_positive=min_positive, vol_weighted=vol_weighted, **kw)
        self.min_positive = int(min_positive)
        self.vol_weighted = bool(vol_weighted)

    def target_weights(self, ctx: AllocatorContext) -> np.ndarray:
        mom = np.asarray(ctx.alpha, dtype=float)
        if not np.all(np.isfinite(mom)):
            return ctx.current_weights.copy()

        signal = (mom > 0).astype(float)
        if signal.sum() < self.min_positive:
            return np.zeros(self.n)

        if self.vol_weighted and ctx.vols is not None:
            sd = np.asarray(ctx.vols, dtype=float)
            sd = np.where(sd > 0, sd, np.nan)
            w = np.nan_to_num(signal / sd, nan=0.0)
        else:
            w = signal
        return w / w.sum() if w.sum() > 0 else np.zeros(self.n)


@register("atr_risk_parity")
class ATRRiskParity(Allocator):
    """Literal Clenow-style position sizing, as a weight.

    The book's formula is `shares = AccountValue * risk_factor / ATR20`,
    i.e. a target dollar-risk-per-position (`risk_factor` of NAV) sized
    inversely to that name's own volatility (ATR). Collapsed to a weight:

        weight_i = risk_factor / atr_pct_i        (atr_pct = ATR / price)

    Unlike `cross_sectional`'s "inverse_vol" weighting, this is NOT
    renormalised to sum to 1 across the selected names. If the raw total
    exceeds 1 it is scaled down (this engine is long-only, unleveraged); if
    it's LESS than 1, the shortfall is deliberately left as cash. That cash
    drag is a real, documented property of this sizing rule when few names
    are held or their volatility is low relative to `risk_factor` -- not a
    bug to normalise away.

    `ctx.vols` here must be each name's ATR expressed as a FRACTION of its
    own price (ATR / price), not an absolute vol number.
    """
    requires = ("alpha", "eligible", "vols")

    def __init__(self, assets, n_hold=None, quantile=None, risk_factor=0.001,
                 max_weight=None, ascending=False, min_names=5, **kw):
        super().__init__(assets, n_hold=n_hold, quantile=quantile, risk_factor=risk_factor,
                         max_weight=max_weight, ascending=ascending, min_names=min_names, **kw)
        if (n_hold is None) == (quantile is None):
            raise ValueError("atr_risk_parity needs exactly one of n_hold or quantile")
        self.n_hold = int(n_hold) if n_hold is not None else None
        self.quantile = float(quantile) if quantile is not None else None
        self.risk_factor = float(risk_factor)
        self.max_weight = float(max_weight) if max_weight is not None else None
        self.ascending = bool(ascending)
        self.min_names = int(min_names)
        self._skipped = 0
        self._held: List[int] = []
        self._cash_shortfall: List[float] = []

    def target_weights(self, ctx: AllocatorContext) -> np.ndarray:
        score = np.asarray(ctx.alpha, dtype=float)
        atr_pct = np.asarray(ctx.vols, dtype=float)
        ok = (np.asarray(ctx.eligible, dtype=bool) & np.isfinite(score)
              & np.isfinite(atr_pct) & (atr_pct > 0))
        n_ok = int(ok.sum())

        if n_ok < self.min_names:
            self._skipped += 1
            return ctx.current_weights.copy()

        k = self.n_hold if self.n_hold is not None else max(1, int(round(self.quantile * n_ok)))
        k = min(k, n_ok)

        idx = np.flatnonzero(ok)
        s = score[idx]
        order = np.argsort(s if self.ascending else -s, kind="stable")
        chosen = idx[order[:k]]
        self._held.append(k)

        if not ctx.buys_allowed:
            return _sell_only(ctx.current_weights, chosen)

        w = np.zeros(self.n)
        w[chosen] = self.risk_factor / atr_pct[chosen]
        if self.max_weight is not None:
            w = np.minimum(w, self.max_weight)
        total = w.sum()
        self._cash_shortfall.append(max(0.0, 1.0 - total))
        if total > 1.0:
            w = w / total
        return w

    def diagnostics(self) -> Dict[str, Any]:
        return {"rebalances_skipped_thin_universe": self._skipped,
                "mean_names_held": (float(np.mean(self._held)) if self._held else 0.0),
                "mean_cash_shortfall_from_sizing": (
                    float(np.mean(self._cash_shortfall)) if self._cash_shortfall else 0.0)}
