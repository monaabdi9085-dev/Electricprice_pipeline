from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pricefc.timeutils import (
    PT15M,
    PT60M,
    aggregate_to_hourly,
    ensure_utc_index,
    expected_periods,
    hours_in_local_day,
    infer_row_resolution,
    local_day_bounds_utc,
)

TZ = "Europe/Stockholm"
days = st.dates(min_value=date(2015, 1, 1), max_value=date(2035, 12, 31))


@pytest.mark.parametrize(
    ("day", "hours"),
    [(date(2025, 3, 30), 23), (date(2025, 10, 26), 25), (date(2025, 6, 1), 24)],
)
def test_hours_in_local_day_dst(day: date, hours: int) -> None:
    assert hours_in_local_day(day, TZ) == hours
    assert expected_periods(day, TZ, PT60M) == hours
    assert expected_periods(day, TZ, PT15M) == 4 * hours


def test_naive_index_rejected() -> None:
    with pytest.raises(ValueError, match="naive"):
        ensure_utc_index(pd.date_range("2025-01-01", periods=3, freq="h"))


def test_infer_resolution_across_mtu_switch() -> None:
    start, switch = local_day_bounds_utc(date(2025, 9, 30), TZ)
    hourly = pd.date_range(start, switch, freq="h", inclusive="left")
    quarter = pd.date_range(switch, periods=96, freq="15min")
    res = infer_row_resolution(hourly.append(quarter))
    assert (res.iloc[: len(hourly)] == PT60M).all()
    assert (res.iloc[len(hourly) :] == PT15M).all()


def test_infer_resolution_marks_row_before_gap_from_previous_spacing() -> None:
    idx = pd.DatetimeIndex(
        ["2025-01-01 00:00", "2025-01-01 01:00", "2025-01-01 05:00", "2025-01-01 06:00"], tz="UTC"
    )
    assert infer_row_resolution(idx).tolist() == [PT60M, PT60M, PT60M, PT60M]


@given(days)
def test_local_day_bounds_round_trip(day: date) -> None:
    start, end = local_day_bounds_utc(day, TZ)
    assert start.tz_convert(TZ).date() == day
    assert start.tz_convert(TZ).hour == 0
    assert end.tz_convert(TZ).date() == day + timedelta(days=1)
    assert (end - start) in {pd.Timedelta(hours=h) for h in (23, 24, 25)}


@given(days, st.integers(min_value=0, max_value=2**32 - 1))
def test_aggregate_quarter_hours_preserves_mean_and_row_count(day: date, seed: int) -> None:
    start, end = local_day_bounds_utc(day, TZ)
    idx = pd.date_range(start, end, freq="15min", inclusive="left")
    values = np.random.default_rng(seed).normal(50, 30, len(idx))
    df = pd.DataFrame({"price": values, "resolution": PT15M}, index=idx)
    out = aggregate_to_hourly(df, ["price"])
    assert len(out) == hours_in_local_day(day, TZ)
    assert (out["n_periods"] == 4).all()
    assert out["price"].mean() == pytest.approx(values.mean())
    assert (out["source_resolution"] == PT15M).all()
