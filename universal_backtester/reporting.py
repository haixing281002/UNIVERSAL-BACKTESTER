"""Turn a BacktestResult's daily weight path into a discrete buy/sell log.

The engine only stores the CONTINUOUS state (post-trade weight per asset per
day) -- it never records "bought X on date Y" as an event. This reconstructs
that event log by diffing each rebalance date's post-trade weights against
the drifted (pre-trade) weights from the day immediately before it, which is
the actual trade the engine executed that day.
"""
from __future__ import annotations

import pandas as pd


def derive_trade_log(result) -> pd.DataFrame:
    """One row per (date, asset) where the engine's weight in that asset
    changed on a rebalance date. `delta` is the change in portfolio weight
    (not shares -- this engine is weight-based, see README's "Design
    boundary" on what "buying an index" means here)."""
    w = result.weights
    idx = w.index
    rows = []
    for d in result.rebalances:
        pos = idx.get_loc(d)
        pre = w.iloc[pos - 1] if pos > 0 else pd.Series(0.0, index=w.columns)
        post = w.loc[d]
        delta = post - pre
        for name in w.columns:
            dw = float(delta[name])
            if abs(dw) > 1e-6:
                rows.append({
                    "date": d, "asset": name,
                    "action": "BUY" if dw > 0 else "SELL",
                    "weight_before": round(float(pre[name]), 6),
                    "weight_after": round(float(post[name]), 6),
                    "delta_weight": round(dw, 6),
                })
    log = pd.DataFrame(rows)
    if not log.empty:
        log = log.sort_values(["date", "asset"]).reset_index(drop=True)
    return log
