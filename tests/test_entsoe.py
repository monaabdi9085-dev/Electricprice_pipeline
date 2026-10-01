from datetime import date
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from entsoe.exceptions import NoMatchingDataError

from pricefc.config import load_config, load_ingest_config
from pricefc.ingest.entsoe import (
    EntsoeFetcher,
    chunk_range,
    jobs_for_zone,
    normalise_timeseries,
    price_job,
    to_entsoe_area,
)
from pricefc.ingest.run import ingest_entsoe
from pricefc.ingest.throttle import RateLimiter
from pricefc.timeutils import PT15M, PT60M, local_day_bounds_utc

CFG = load_ingest_config(Path("configs/ingest.yaml"))
TZ = "Europe/Stockholm"
SWITCH = local_day_bounds_utc(date(2025, 10, 1), TZ)[0]


class FakeEntsoe:
    """Mimics entsoe-py: local-tz index, inclusive `end`, 15-min MTUs from 2025-10-01."""

    def __init__(self, empty: bool = False) -> None:
        self.calls: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
        self.empty = empty

    def query_day_ahead_prices(
        self, country_code: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series:
        self.calls.append((country_code, start, end))
        if self.empty:
            raise NoMatchingDataError
        hourly = pd.date_range(start, min(end, SWITCH), freq="h", inclusive="left")
        quarter = pd.date_range(max(start, SWITCH), end, freq="15min") if end > SWITCH else []
        idx = hourly.append(pd.DatetimeIndex(quarter)).tz_convert(TZ)
        return pd.Series(np.linspace(-10, 200, len(idx)), index=idx)

    def query_generation(
        self, country_code: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        idx = pd.date_range(start, end, freq="h", inclusive="left").tz_convert(TZ)
        cols = pd.MultiIndex.from_tuples(
            [("Nuclear", "Actual Aggregated"), ("Hydro Water Reservoir", "Actual Aggregated")]
        )
        return pd.DataFrame(1000.0, index=idx, columns=cols)


def fetcher(client: FakeEntsoe, chunk_days: int = 30) -> EntsoeFetcher:
    cfg = CFG.entsoe.model_copy(update={"chunk_days": chunk_days})
    return EntsoeFetcher(cfg, client, limiter=RateLimiter(0))  # type: ignore[arg-type]


def test_area_codes() -> None:
    assert to_entsoe_area("SE3") == "SE_3"
    assert to_entsoe_area("DE_LU") == "DE_LU"
    assert to_entsoe_area("FI") == "FI"


def test_chunk_range_covers_without_overlap() -> None:
    s, e = pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2023-03-01", tz="UTC")
    chunks = chunk_range(s, e, 365)
    assert chunks[0][0] == s and chunks[-1][1] == e
    assert all(a[1] == b[0] for a, b in pairwise(chunks))


def test_jobs_for_se3() -> None:
    jobs = jobs_for_zone("SE3", CFG.entsoe)
    keys = {(j.dataset, j.key) for j in jobs}
    assert ("day_ahead_prices", "SE3") in keys
    assert ("hydro_reservoirs", "SE") in keys
    assert ("crossborder_flows", "SE3_to_FI") in keys
    assert ("crossborder_flows", "NO1_to_SE3") in keys
    assert ("scheduled_exchanges", "SE3_to_SE4") in keys


def test_prices_across_mtu_switch_chunked_utc_no_overlap() -> None:
    client = FakeEntsoe()
    pull = fetcher(client).fetch(price_job("SE3"), date(2025, 9, 1), date(2025, 11, 1))
    assert len(client.calls) == 3
    df = pull.data
    assert str(df["timestamp"].dt.tz) == "UTC"
    assert df["timestamp"].is_unique and df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].max() < pd.Timestamp("2025-11-01", tz="UTC")
    assert (df.loc[df["timestamp"] < SWITCH, "resolution"] == PT60M).all()
    assert (df.loc[df["timestamp"] >= SWITCH, "resolution"] == PT15M).all()


def test_multiindex_columns_flattened() -> None:
    idx = pd.date_range("2025-01-01", periods=2, freq="h", tz=TZ)
    raw = FakeEntsoe().query_generation("SE_3", idx[0], idx[-1] + pd.Timedelta("1h"))
    df = normalise_timeseries(raw, "value")
    assert list(df.columns) == [
        "timestamp",
        "nuclear__actual_aggregated",
        "hydro_water_reservoir__actual_aggregated",
    ]


@pytest.mark.parametrize("empty", [False, True])
def test_ingest_entsoe_end_to_end(tmp_path: Path, empty: bool) -> None:
    base = load_config(
        Path("configs/base.yaml"),
        {
            "paths": {
                "data_root": str(tmp_path),
                "raw": str(tmp_path / "raw"),
                "datasets": str(tmp_path / "ds"),
            }
        },
    )
    [result] = ingest_entsoe(
        base,
        CFG,
        [price_job("SE3")],
        date(2025, 9, 25),
        date(2025, 10, 5),
        client=FakeEntsoe(empty=empty),
    )  # type: ignore[arg-type]
    assert result.report.passed is (not empty), result.report.errors
    assert result.snapshot.manifest["known_limitations"]
