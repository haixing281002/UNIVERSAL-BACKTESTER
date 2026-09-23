"""Generic data loaders. Each returns a tidy wide frame (DatetimeIndex x
series) plus a provenance dict (source path, sha256, per-series coverage) --
so a result can always be traced back to the exact bytes it came from.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Tuple

import pandas as pd


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_wide_csv(path: str, date_col: str = "Date") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load a plain wide CSV: one date column, one column per asset's price."""
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df.index.name = "date"
    prov = {
        "loader": "load_wide_csv", "path": path, "sha256": file_sha256(path),
        "n_rows": int(len(df)), "n_series": int(df.shape[1]),
        "date_min": str(df.index.min().date()), "date_max": str(df.index.max().date()),
    }
    return df, prov


def load_banner_workbook(path: str, sheet="Sheet1") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load a workbook laid out as: a banner row of series names, a row of
    field names (one of which is 'Date'), a shared date column, and blank
    spacer columns between series blocks. Common export format for index /
    benchmark data providers.

    The parser locates the date column and the name/field rows by content
    rather than fixed offsets, so a re-export with shifted columns still
    loads. If a series has multiple fields (Open/High/Low/Close/...), all are
    kept, prefixed as "SeriesName Field"; a lone 'Close'-like field is kept
    bare as "SeriesName".
    """
    raw = pd.read_excel(path, sheet_name=sheet, header=None)

    date_row = date_col = None
    for r in range(min(12, len(raw))):
        for c in range(raw.shape[1]):
            v = raw.iat[r, c]
            if isinstance(v, str) and v.strip().lower() == "date":
                date_row, date_col = r, c
                break
        if date_row is not None:
            break
    if date_row is None:
        raise ValueError(f"{path}: could not locate a 'Date' header cell")

    name_row = date_row - 1
    dates = pd.to_datetime(raw.iloc[date_row + 1:, date_col], errors="coerce")

    # group columns into contiguous blocks under each name
    blocks: Dict[str, list] = {}
    current_name = None
    for c in range(raw.shape[1]):
        if c == date_col:
            continue
        nm = raw.iat[name_row, c] if name_row >= 0 else None
        if isinstance(nm, str) and nm.strip():
            current_name = re.sub(r"\s+", " ", nm.strip())
            blocks.setdefault(current_name, [])
        fld = raw.iat[date_row, c]
        # A column with no real field label (e.g. a spacer that happens to
        # carry a stray value) must NOT be defaulted to a guessed field name
        # -- an earlier version of this defaulted unlabeled columns to
        # "Close", which let a near-empty stray spacer column silently
        # overwrite a real Close series later (same dict key, last write
        # wins). Skip anything without a genuine string field label instead.
        if not (isinstance(fld, str) and fld.strip()):
            continue
        vals = pd.to_numeric(raw.iloc[date_row + 1:, c], errors="coerce")
        if vals.notna().sum() == 0 or current_name is None:
            continue
        blocks[current_name].append((fld.strip(), vals))

    series: Dict[str, pd.Series] = {}
    for name, fields in blocks.items():
        # A duplicate field label under the same name (e.g. two genuine
        # "Close" columns) is a data-layout surprise, not something to
        # silently pick a winner for.
        labels = [f for f, _ in fields]
        if len(labels) != len(set(labels)):
            dupes = sorted({l for l in labels if labels.count(l) > 1})
            raise ValueError(f"{path} [{sheet}]: series '{name}' has duplicate "
                             f"field label(s) {dupes} -- cannot tell which column is which")
        if len(fields) == 1:
            fld, vals = fields[0]
            series[name] = pd.Series(vals.values, index=dates.values)
        else:
            for fld, vals in fields:
                series[f"{name} {fld}"] = pd.Series(vals.values, index=dates.values)

    if not series:
        raise ValueError(f"{path}: no numeric series found")

    df = pd.DataFrame(series)
    df.index.name = "date"
    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]

    prov = {
        "loader": "load_banner_workbook", "path": path, "sha256": file_sha256(path),
        "sheet": str(sheet), "n_rows": int(len(df)), "n_series": int(df.shape[1]),
        "date_min": str(df.index.min().date()), "date_max": str(df.index.max().date()),
        "series": {
            name: {
                "first": str(df[name].dropna().index.min().date()) if df[name].notna().any() else None,
                "last": str(df[name].dropna().index.max().date()) if df[name].notna().any() else None,
                "n_obs": int(df[name].notna().sum()),
            } for name in df.columns
        },
    }
    return df, prov


_KNOWN_FIELDS = ("Open", "High", "Low", "Close", "PE", "PB")


def load_field_only(path: str, field: str = "Close", sheets=None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load one or more sheets of a `load_banner_workbook`-shaped file, keep
    only each series' `field` column (default "Close"), and return one
    column per series named plainly (no " Close"/" High"/... suffix) -- one
    price (or High, or Low, ...) per tradable name.

    `sheets=None` autodetects every sheet in the workbook. Call this once
    per field you need (e.g. once for "Close", once for "High", once for
    "Low" to build an ATR) -- each call re-reads the workbook, which is
    cheap relative to actually computing anything on the result.
    """
    import openpyxl
    if sheets is None:
        wb = openpyxl.load_workbook(path, read_only=True)
        sheets = wb.sheetnames
        wb.close()

    suffix = f" {field}"
    frames = []
    prov = {"loader": "load_field_only", "field": field, "path": path,
            "sha256": file_sha256(path), "sheets": {}}
    for sh in sheets:
        df, sub_prov = load_banner_workbook(path, sheet=sh)
        matched = {c: c[:-len(suffix)] for c in df.columns if c.endswith(suffix)}
        # A series with only ONE field total (no OHLC breakdown) keeps its
        # bare name in load_banner_workbook -- only usable as "Close".
        bare_cols = {}
        if field == "Close":
            bare_cols = {c: c for c in df.columns if " " not in c or not any(
                c.endswith(f" {f}") for f in _KNOWN_FIELDS)}
        keep = {**matched, **bare_cols}
        if keep:
            frames.append(df[list(keep.keys())].rename(columns=keep))
        prov["sheets"][sh] = sub_prov

    if not frames:
        raise ValueError(f"{path}: no series had a '{field}' field across sheets {sheets}")
    merged = pd.concat(frames, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated(keep="first")]
    prov["n_series"] = int(merged.shape[1])
    prov["date_min"] = str(merged.index.min().date())
    prov["date_max"] = str(merged.index.max().date())
    return merged, prov


def load_close_only(path: str, sheets=None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Backward-compatible alias: `load_field_only(path, field="Close", ...)`."""
    return load_field_only(path, field="Close", sheets=sheets)


def audit_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Per-series data-quality audit -- run this before any backtest touches
    the data. Flags exactly the things that produce a normal-looking backtest
    that is silently wrong: gaps inside coverage, suspicious flat stretches,
    implausible single-day moves, non-positive prices."""
    rows = []
    for c in df.columns:
        s = df[c].dropna()
        if s.empty:
            rows.append({"series": c, "n_obs": 0})
            continue
        r = s.pct_change().dropna()
        inside = df[c].loc[s.index.min():s.index.max()]
        rows.append({
            "series": c, "n_obs": len(s),
            "first": s.index.min().date(), "last": s.index.max().date(),
            "gaps_inside_coverage": int(inside.isna().sum()),
            "max_abs_daily_return": round(float(r.abs().max()), 4) if len(r) else float("nan"),
            "n_daily_moves_gt_20pct": int((r.abs() > 0.20).sum()),
            "n_stale_5day_runs": int((r.rolling(5).apply(lambda x: (x == 0).all(), raw=True) == 1).sum()),
            "non_positive_prices": int((s <= 0).sum()),
        })
    return pd.DataFrame(rows)
