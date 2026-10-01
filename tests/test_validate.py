from datetime import date

import pandas as pd
import pytest

from pricefc.timeutils import PT15M, PT60M, infer_row_resolution, local_day_bounds_utc
from pricefc.validate.schemas import validate_series
from pricefc.validate.specs import entsoe_spec, weather_spec

TZ = "Europe/Stockholm"


def prices(
    first: date, last: date, freq: str = "h", value: float = 50.0
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    start, _ = local_day_bounds_utc(first, TZ)
    _, end = local_day_bounds_utc(last, TZ)
    ts = pd.date_range(start, end, freq=freq, inclusive="left")
    df = pd.DataFrame({"timestamp": ts, "price_eur_mwh": value})
    df["resolution"] = infer_row_resolution(pd.DatetimeIndex(ts)).to_numpy()
    return df, start, end


def run(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):  # type: ignore[no-untyped-def]
    return validate_series(
        df, entsoe_spec("day_ahead_prices"), tz=TZ, requested_start=start, requested_end=end
    )


@pytest.mark.parametrize("day", [date(2025, 3, 30), date(2025, 10, 26), date(2024, 6, 1)])
def test_valid_hourly_prices_including_dst_days(day: date) -> None:
    report = run(*prices(day, day))
    assert report.passed, report.errors


def test_valid_across_mtu_switch() -> None:
    a, start, _ = prices(date(2025, 9, 29), date(2025, 9, 30))
    b, _, end = prices(date(2025, 10, 1), date(2025, 10, 2), freq="15min")
    df = pd.concat([a, b], ignore_index=True)
    df["resolution"] = infer_row_resolution(pd.DatetimeIndex(df["timestamp"])).to_numpy()
    report = run(df, start, end)
    assert report.passed, report.errors
    assert report.stats["resolution_counts"] == {PT60M: 48, PT15M: 192}


def test_missing_hour_fails_day_count_and_gap() -> None:
    df, start, end = prices(date(2025, 1, 10), date(2025, 1, 10))
    report = run(df.drop(index=5).reset_index(drop=True), start, end)
    checks = {e["check"] for e in report.errors}
    assert not report.passed
    assert {"day_count", "max_gap"} <= checks


def test_duplicate_and_naive_timestamps_fail() -> None:
    df, start, end = prices(date(2025, 1, 10), date(2025, 1, 10))
    dup = pd.concat([df, df.iloc[[3]]]).sort_values("timestamp").reset_index(drop=True)
    assert not run(dup, start, end).passed
    naive = df.assign(timestamp=df["timestamp"].dt.tz_localize(None))
    assert not run(naive, start, end).passed


def test_negative_price_is_flagged_not_rejected() -> None:
    df, start, end = prices(date(2025, 1, 10), date(2025, 1, 10), value=-5.0)
    report = run(df, start, end)
    assert report.passed
    assert report.warnings


def test_out_of_range_price_rejected() -> None:
    df, start, end = prices(date(2025, 1, 10), date(2025, 1, 10), value=50_000.0)
    assert not run(df, start, end).passed


def test_weather_nulls_and_ranges() -> None:
    ts = pd.date_range("2025-01-10", "2025-01-11", freq="h", tz="UTC", inclusive="left")
    df = pd.DataFrame({"timestamp": ts, "temperature_2m": 1.0, "cloud_cover_previous_day1": 50.0})
    spec = weather_spec("w", list(df.columns))
    kwargs = {"tz": "UTC", "requested_start": ts[0], "requested_end": ts[-1] + pd.Timedelta("1h")}
    assert validate_series(df, spec, **kwargs).passed  # type: ignore[arg-type]
    bad = df.copy()
    bad.loc[3, "temperature_2m"] = None
    bad.loc[4, "cloud_cover_previous_day1"] = 140.0
    report = validate_series(bad, spec, **kwargs)  # type: ignore[arg-type]
    assert not report.passed
    assert {c for e in report.errors for c in e.get("columns", [])} == {
        "temperature_2m",
        "cloud_cover_previous_day1",
    }


def test_unknown_weather_column_has_no_range() -> None:
    with pytest.raises(KeyError):
        weather_spec("w", ["timestamp", "mystery"])


def test_slightly_negative_radiation_is_flagged_not_rejected() -> None:
    ts = pd.date_range("2025-10-28", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "shortwave_radiation_previous_day2": 0.0})
    df.loc[20, "shortwave_radiation_previous_day2"] = -1.0
    report = validate_series(
        df,
        weather_spec("w", list(df.columns)),
        tz="UTC",
        requested_start=ts[0],
        requested_end=ts[-1] + pd.Timedelta("1h"),
    )
    assert report.passed and report.warnings
