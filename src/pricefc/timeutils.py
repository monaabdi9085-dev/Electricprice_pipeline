"""Time handling: UTC storage, local-day arithmetic, DST and market resolution.

All stored timestamps are UTC. Local time (Europe/Stockholm) is derived only where a
local-calendar concept is needed (delivery days, calendar features).
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

UTC = "UTC"
PT60M = "PT60M"
PT15M = "PT15M"
MIXED = "mixed"
RESOLUTION_STEPS = {PT60M: pd.Timedelta(hours=1), PT15M: pd.Timedelta(minutes=15)}
# SDAC switched to 15-minute market time units for delivery from this local day onwards.
MTU15_GO_LIVE = date(2025, 10, 1)


def ensure_utc_index(index: pd.Index) -> pd.DatetimeIndex:
    """Convert a tz-aware DatetimeIndex to UTC; reject naive timestamps."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"expected DatetimeIndex, got {type(index).__name__}")
    if index.tz is None:
        raise ValueError("naive timestamps are not allowed; localise before converting to UTC")
    return index.tz_convert(UTC)


def local_day_bounds_utc(day: date, tz: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """UTC [start, end) of a local calendar day. Handles 23- and 25-hour DST days."""
    start = pd.Timestamp(day).tz_localize(tz)
    end = pd.Timestamp(day + timedelta(days=1)).tz_localize(tz)
    return start.tz_convert(UTC), end.tz_convert(UTC)


def hours_in_local_day(day: date, tz: str) -> int:
    start, end = local_day_bounds_utc(day, tz)
    return int((end - start) / pd.Timedelta(hours=1))


def expected_periods(day: date, tz: str, resolution: str) -> int:
    """Number of market time units in a local delivery day at the given resolution."""
    return int(hours_in_local_day(day, tz) * (pd.Timedelta(hours=1) / RESOLUTION_STEPS[resolution]))


def infer_row_resolution(index: pd.DatetimeIndex) -> pd.Series:
    """Per-row resolution from the spacing to the next timestamp (last row: previous spacing).

    Works across the hourly -> 15-minute switch because it is evaluated per row.
    """
    if len(index) == 0:
        return pd.Series([], index=index, dtype="string")
    if len(index) == 1:
        raise ValueError("cannot infer resolution from a single timestamp")
    fwd = pd.Series(index, index=index).diff().shift(-1)
    fwd.iloc[-1] = fwd.iloc[-2]
    out = pd.Series(pd.NA, index=index, dtype="string")
    for label, step in RESOLUTION_STEPS.items():
        out[fwd == step] = label
    # Rows before a gap: fall back to the spacing from the previous row.
    back = pd.Series(index, index=index).diff()
    for label, step in RESOLUTION_STEPS.items():
        out[out.isna() & (back == step)] = label
    return out


def aggregate_to_hourly(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Mean-aggregate sub-hourly rows to UTC hours.

    Input: UTC DatetimeIndex and a `resolution` column. Output adds `n_periods` (rows that
    contributed) and `source_resolution` (PT60M, PT15M or mixed). Stockholm's UTC offset is a
    whole number of hours, so UTC hours coincide with local clock hours.
    """
    idx = ensure_utc_index(df.index)
    hour = idx.floor("h")
    grouped = df.groupby(hour)
    out = grouped[value_cols].mean()
    out["n_periods"] = grouped.size()
    res = grouped["resolution"].agg(lambda s: s.iloc[0] if s.nunique() == 1 else MIXED)
    out["source_resolution"] = res.astype("string")
    out.index.name = df.index.name
    return out
