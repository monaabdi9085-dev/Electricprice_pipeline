"""ENTSO-E Transparency Platform ingestion via entsoe-py (spec section 5.1).

The API token is read from ENTSOE_API_TOKEN by the caller. Requests are chunked (the API caps
the date range per request), throttled and retried. All timestamps are converted to UTC and
each row gets a `resolution` (PT60M/PT15M) inferred from the series spacing, so data from
before and after the 2025-10-01 switch to 15-minute MTUs is handled explicitly.

Known limitation: forecast series have no issue-time vintages; a backfill returns the latest
published values, which may include revisions. `pulled_at` is recorded in every manifest.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

import pandas as pd
import requests

from pricefc.config import EntsoeConfig
from pricefc.ingest.throttle import RateLimiter, RetryableError, with_retries
from pricefc.timeutils import infer_row_resolution

SOURCE = "entsoe"
PER_NEIGHBOUR = ("crossborder_flows", "scheduled_exchanges")
EVENT_DATASETS = ("generation_unavailability",)


class EntsoeClient(Protocol):
    """The subset of entsoe.EntsoePandasClient used here (allows fakes in tests)."""

    def query_day_ahead_prices(
        self, country_code: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series: ...
    def query_load_forecast(
        self, country_code: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame: ...
    def query_wind_and_solar_forecast(
        self, country_code: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame: ...
    def query_generation(
        self, country_code: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame: ...
    def query_crossborder_flows(
        self, country_code_from: str, country_code_to: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series: ...
    def query_scheduled_exchanges(
        self,
        country_code_from: str,
        country_code_to: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        dayahead: bool = False,
    ) -> pd.Series: ...
    def query_unavailability_of_generation_units(
        self, country_code: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame: ...
    def query_aggregate_water_reservoirs_and_hydro_storage(
        self, country_code: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame: ...


def to_entsoe_area(name: str) -> str:
    """Config zone names (SE3, NO1, DK2, DE_LU, FI) -> entsoe-py area codes (SE_3, ...)."""
    m = re.fullmatch(r"([A-Z]{2})(\d)", name)
    return f"{m.group(1)}_{m.group(2)}" if m else name


def from_entsoe_area(code: str) -> str:
    m = re.fullmatch(r"([A-Z]{2})_(\d)", code)
    return f"{m.group(1)}{m.group(2)}" if m else code


def neighbours(zone: str) -> list[str]:
    from entsoe.mappings import NEIGHBOURS

    # DE_AT_LU is the pre-2018 German zone; outside our history window.
    return [from_entsoe_area(n) for n in NEIGHBOURS[to_entsoe_area(zone)] if n != "DE_AT_LU"]


def snake(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def chunk_range(
    start: pd.Timestamp, end: pd.Timestamp, days: int
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split [start, end) into consecutive windows of at most `days` days."""
    if end <= start:
        raise ValueError("end must be after start")
    out = []
    cur = start
    step = pd.Timedelta(days=days)
    while cur < end:
        nxt = min(cur + step, end)
        out.append((cur, nxt))
        cur = nxt
    return out


def normalise_timeseries(obj: pd.Series | pd.DataFrame, value_name: str) -> pd.DataFrame:
    """entsoe-py output -> DataFrame[timestamp (UTC), <snake_case value cols>]."""
    if isinstance(obj, pd.Series):
        df = obj.to_frame(value_name)
    else:
        df = obj.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["__".join(snake(p) for p in col if str(p)) for col in df.columns]
        else:
            df.columns = [snake(c) for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError("expected a tz-aware DatetimeIndex from entsoe-py")
    df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    df = df.reset_index()
    value_cols = [c for c in df.columns if c != "timestamp"]
    return df.astype({c: "float64" for c in value_cols})


def normalise_events(df: pd.DataFrame) -> pd.DataFrame:
    """Unavailability tables: all datetimes to UTC; `timestamp` = event start."""
    out = df.reset_index()
    out.columns = [snake(c) for c in out.columns]
    for col in out.columns:
        if isinstance(out[col].dtype, pd.DatetimeTZDtype):
            out[col] = out[col].dt.tz_convert("UTC")
    if "start" not in out.columns:
        raise ValueError(f"unavailability table has no 'start' column: {list(out.columns)}")
    out.insert(0, "timestamp", out.pop("start"))
    out = out.sort_values("timestamp", kind="stable").reset_index(drop=True)
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype("string")
    return out


@dataclass
class EntsoeJob:
    dataset: str
    key: str
    query: dict[str, Any]
    call: Callable[[EntsoeClient, pd.Timestamp, pd.Timestamp], pd.Series | pd.DataFrame]
    value_name: str = "value"
    is_events: bool = False


@dataclass
class EntsoePull:
    job: EntsoeJob
    data: pd.DataFrame
    requested_start: pd.Timestamp
    requested_end: pd.Timestamp
    chunks: list[dict[str, Any]] = field(default_factory=list)


# Single-area datasets: client method, value column name (for Series output), event table?
_AREA_METHODS: dict[str, tuple[str, str, bool]] = {
    "day_ahead_prices": ("query_day_ahead_prices", "price_eur_mwh", False),
    "load_forecast": ("query_load_forecast", "value", False),
    "wind_solar_forecast": ("query_wind_and_solar_forecast", "value", False),
    "generation": ("query_generation", "value", False),
    "generation_unavailability": ("query_unavailability_of_generation_units", "value", True),
    "hydro_reservoirs": (
        "query_aggregate_water_reservoirs_and_hydro_storage",
        "stored_energy_mwh",
        False,
    ),
}


def area_job(dataset: str, area_name: str) -> EntsoeJob:
    method, value_name, is_events = _AREA_METHODS[dataset]
    area = to_entsoe_area(area_name)

    def call(c: EntsoeClient, s: pd.Timestamp, e: pd.Timestamp) -> pd.Series | pd.DataFrame:
        result: pd.Series | pd.DataFrame = getattr(c, method)(area, s, e)
        return result

    query = {"area": area, "method": method}
    if dataset == "day_ahead_prices":
        query["document_type"] = "A44"
    return EntsoeJob(dataset, area_name, query, call, value_name=value_name, is_events=is_events)


def price_job(area_name: str) -> EntsoeJob:
    return area_job("day_ahead_prices", area_name)


def jobs_for_zone(zone: str, cfg: EntsoeConfig) -> list[EntsoeJob]:
    jobs: list[EntsoeJob] = []
    for ds in cfg.datasets:
        if ds == "hydro_reservoirs":
            jobs.append(area_job(ds, cfg.hydro_reservoir_area))
        elif ds in PER_NEIGHBOUR:
            for nb in neighbours(zone):
                for src, dst in ((zone, nb), (nb, zone)):
                    jobs.append(_exchange_job(ds, src, dst))
        else:
            jobs.append(area_job(ds, zone))
    return jobs


def _exchange_job(ds: str, src: str, dst: str) -> EntsoeJob:
    a, b = to_entsoe_area(src), to_entsoe_area(dst)

    def call(c: EntsoeClient, s: pd.Timestamp, e: pd.Timestamp) -> pd.Series:
        if ds == "crossborder_flows":
            return c.query_crossborder_flows(a, b, s, e)
        return c.query_scheduled_exchanges(a, b, s, e, dayahead=True)

    return EntsoeJob(ds, f"{src}_to_{dst}", {"from": a, "to": b}, call, value_name="mw")


class EntsoeFetcher:
    def __init__(
        self, cfg: EntsoeConfig, client: EntsoeClient, limiter: RateLimiter | None = None
    ) -> None:
        self.cfg = cfg
        self.client = client
        self.limiter = limiter or RateLimiter(cfg.min_interval_s)

    def _call(self, job: EntsoeJob, s: pd.Timestamp, e: pd.Timestamp) -> Any:
        from entsoe.exceptions import NoMatchingDataError

        def attempt() -> Any:
            self.limiter.wait()
            try:
                return job.call(self.client, s, e)
            except NoMatchingDataError:
                return None
            except requests.HTTPError as err:
                status = err.response.status_code if err.response is not None else None
                if status is None or status == 429 or status >= 500:
                    raise RetryableError(str(err)) from err
                raise
            except (requests.ConnectionError, requests.Timeout) as err:
                raise RetryableError(str(err)) from err

        return with_retries(attempt, max_retries=self.cfg.max_retries)

    def fetch(self, job: EntsoeJob, start: date, end: date) -> EntsoePull:
        """Fetch [start, end) (UTC dates), chunked by `chunk_days`."""
        req_start = pd.Timestamp(start, tz="UTC")
        req_end = pd.Timestamp(end, tz="UTC")
        frames, chunks = [], []
        for s, e in chunk_range(req_start, req_end, self.cfg.chunk_days):
            raw = self._call(job, s, e)
            if raw is None or len(raw) == 0:
                chunks.append({"start": s.isoformat(), "end": e.isoformat(), "rows": 0})
                continue
            if job.is_events:
                df = normalise_events(raw)
            else:
                df = normalise_timeseries(raw, job.value_name)
                # entsoe-py truncates inclusively at `end`; keep [s, e) so chunks don't overlap.
                df = df[(df["timestamp"] >= s) & (df["timestamp"] < e)]
            chunks.append({"start": s.isoformat(), "end": e.isoformat(), "rows": len(df)})
            frames.append(df)
        data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({"timestamp": []})
        if not job.is_events and len(data):
            data = (
                data.drop_duplicates("timestamp", keep="last")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            if len(data) >= 2:
                data["resolution"] = infer_row_resolution(
                    pd.DatetimeIndex(data["timestamp"])
                ).to_numpy()
        return EntsoePull(job, data, req_start, req_end, chunks)
