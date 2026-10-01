"""Smoke tests against the real APIs: one small request per dataset.

Skipped by default (`-m 'not live'`); run with `make test-live`. They catch API changes
(schemas, URL formats, row counts) that recorded fixtures cannot. ENTSO-E tests need
ENTSOE_API_TOKEN.
"""

import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from dotenv import load_dotenv

from pricefc.config import Location, load_ingest_config
from pricefc.ingest.entsoe import EntsoeFetcher, jobs_for_zone
from pricefc.ingest.openmeteo import OpenMeteoClient
from pricefc.ingest.prices import ElprisSource
from pricefc.timeutils import expected_periods
from pricefc.validate.schemas import validate_series
from pricefc.validate.specs import entsoe_spec, weather_spec

pytestmark = pytest.mark.live
load_dotenv()

CFG = load_ingest_config(Path("configs/ingest.yaml"))
TZ = "Europe/Stockholm"
STOCKHOLM = Location(name="se3_stockholm", lat=59.33, lon=18.07, role="demand")
# A fixed, fully published day after the 15-minute switch.
DAY = date(2026, 9, 1)
needs_entsoe = pytest.mark.skipif(
    not os.environ.get("ENTSOE_API_TOKEN"), reason="ENTSOE_API_TOKEN not set"
)


@pytest.mark.parametrize("endpoint", ["historical_forecast", "previous_runs"])
def test_open_meteo_one_day(endpoint: str) -> None:
    client = OpenMeteoClient(CFG.open_meteo)
    pull = client.fetch_range(endpoint, STOCKHOLM, DAY, DAY)  # type: ignore[arg-type]
    assert len(pull.data) == 24
    assert set(client.hourly_variables(endpoint)) <= set(pull.data.columns)
    spec = weather_spec(pull.dataset, list(pull.data.columns))
    report = validate_series(
        pull.data,
        spec,
        tz="UTC",
        requested_start=pull.requested_start,
        requested_end=pull.requested_end,
    )
    assert report.passed, report.errors


@pytest.mark.parametrize("zone", ["SE1", "SE2", "SE3", "SE4"])
def test_elprisetjustnu_one_day(zone: str) -> None:
    pull = ElprisSource(CFG.elprisetjustnu).fetch_prices(zone, DAY, DAY, TZ)
    assert len(pull.data) == expected_periods(DAY, TZ, "PT15M") == 96
    report = validate_series(
        pull.data,
        entsoe_spec("day_ahead_prices"),
        tz=TZ,
        requested_start=pull.requested_start,
        requested_end=pull.requested_end,
    )
    assert report.passed, report.errors


@needs_entsoe
@pytest.mark.parametrize(
    "job", jobs_for_zone("SE3", CFG.entsoe), ids=lambda j: f"{j.dataset}-{j.key}"
)
def test_entsoe_one_day(job) -> None:  # type: ignore[no-untyped-def]
    from pricefc.ingest.run import make_entsoe_client

    fetcher = EntsoeFetcher(CFG.entsoe, make_entsoe_client())
    # Outages and weekly hydro data are sparse; use a wider window for those.
    days = 14 if job.dataset in ("generation_unavailability", "hydro_reservoirs") else 1
    pull = fetcher.fetch(job, DAY, DAY + timedelta(days=days))
    assert len(pull.data) > 0, f"no data for {job.dataset}/{job.key}"
    assert isinstance(pull.data["timestamp"].dtype, pd.DatetimeTZDtype)
    if job.dataset == "day_ahead_prices":
        assert len(pull.data) == 96
        report = validate_series(
            pull.data,
            entsoe_spec(job.dataset),
            tz="UTC",
            requested_start=pull.requested_start,
            requested_end=pull.requested_end,
        )
        assert report.passed, report.errors
