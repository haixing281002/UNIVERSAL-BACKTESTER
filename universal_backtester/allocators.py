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
    vols: Optional[np.ndarray] = None        # per-asset trailing vol
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
