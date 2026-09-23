"""Performance metrics computed from a BacktestResult."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

ANN = 252


def cagr(value: pd.Series, ann: int = ANN) -> float:
    v = value.dropna()
    if len(v) < 2 or v.iloc[0] <= 0:
        return float("nan")
    years = (v.index[-1] - v.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float((v.iloc[-1] / v.iloc[0]) ** (1.0 / years) - 1.0)


def ann_vol(returns: pd.Series, ann: int = ANN) -> float:
    r = returns.dropna()
    return float(r.std(ddof=1) * np.sqrt(ann)) if len(r) > 2 else float("nan")


def sharpe(value: pd.Series, returns: pd.Series, rf_daily: Optional[pd.Series] = None,
           ann: int = ANN) -> float:
    """Arithmetic-mean-excess-return / std, annualised (the conventional definition)."""
    r = returns.dropna()
    if len(r) < 3:
        return float("nan")
    ex = r - (rf_daily.reindex(r.index).fillna(0.0) if rf_daily is not None else 0.0)
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(ann)) if sd > 0 else float("nan")


def drawdown_series(value: pd.Series) -> pd.Series:
    return value / value.cummax() - 1.0


def max_drawdown(value: pd.Series) -> float:
    return float(-drawdown_series(value).min())


def ann_turnover(turnover: pd.Series, ann: int = ANN) -> float:
    days = (turnover.index[-1] - turnover.index[0]).days
    years = days / 365.25 if days > 0 else float("nan")
    return float(turnover.sum() / years) if years and years > 0 else float("nan")


@dataclass
class Metrics:
    name: str
    cagr: float
    vol: float
    sharpe: float
    max_dd: float
    turnover: float
    n_obs: int
    start: str = ""
    end: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_metrics(result, rf_daily: Optional[pd.Series] = None, ann: int = ANN) -> Metrics:
    v, r = result.value, result.returns
    return Metrics(
        name=result.name,
        cagr=cagr(v, ann),
        vol=ann_vol(r, ann),
        sharpe=sharpe(v, r, rf_daily, ann),
        max_dd=max_drawdown(v),
        turnover=ann_turnover(result.turnover, ann),
        n_obs=int(len(v)),
        start=str(v.index[0].date()), end=str(v.index[-1].date()),
    )


def metrics_table(results, rf_daily: Optional[pd.Series] = None) -> pd.DataFrame:
    rows = [compute_metrics(r, rf_daily).to_dict() for r in results]
    return pd.DataFrame(rows).set_index("name")


def render_table(df: pd.DataFrame) -> str:
    out = df.copy()
    for c in ["cagr", "vol", "max_dd", "turnover"]:
        if c in out.columns:
            out[c] = out[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "--")
    if "sharpe" in out.columns:
        out["sharpe"] = out["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "--")
    return out.to_string()
