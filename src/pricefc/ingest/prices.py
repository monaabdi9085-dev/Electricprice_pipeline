"""Pluggable day-ahead price sources.

Every source returns the same table: timestamp (UTC, interval start), price_eur_mwh and
resolution (PT60M/PT15M), plus source-specific raw columns. The snapshot `source` field
records which one produced a series, so ENTSO-E and elprisetjustnu.se can be compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Protocol

import pandas as pd
import requests

from pricefc.config import ElprisConfig, EntsoeConfig
from pricefc.ingest.throttle import RateLimiter, RetryableError, with_retries
from pricefc.timeutils import infer_row_resolution, local_day_bounds_utc

PRICE_COLUMNS = ("timestamp", "price_eur_mwh", "resolution")


@dataclass
class PricePull:
    source: str
    zone: str
    data: pd.DataFrame
    requested_start: pd.Timestamp
    requested_end: pd.Timestamp
    query: dict[str, Any]
    endpoint: str
    chunks: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class PriceSource(Protocol):
    name: str

    def fetch_prices(self, zone: str, first: date, last: date, tz: str) -> PricePull:
        """Prices for local delivery days first..last (inclusive)."""
        ...


def month_ranges(first: date, last: date) -> list[tuple[date, date]]:
    """Split an inclusive date range into inclusive calendar-month ranges."""
    out = []
    cur = first
    while cur <= last:
        nxt = date(cur.year + cur.month // 12, cur.month % 12 + 1, 1)
        out.append((cur, min(nxt - timedelta(days=1), last)))
        cur = nxt
    return out


class ElprisSource:
    """elprisetjustnu.se: one JSON file per local day and zone (no key)."""

    name = "elprisetjustnu"

    def __init__(
        self,
        cfg: ElprisConfig,
        session: requests.Session | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        self.cfg = cfg
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = cfg.user_agent
        self.limiter = limiter or RateLimiter(cfg.min_interval_s)

    def url(self, zone: str, day: date) -> str:
        return self.cfg.url_template.format(year=f"{day:%Y}", month_day=f"{day:%m-%d}", zone=zone)

    def _get_day(self, zone: str, day: date) -> list[dict[str, Any]] | None:
        url = self.url(zone, day)

        def call() -> list[dict[str, Any]] | None:
            self.limiter.wait()
            try:
                resp = self.session.get(url, timeout=self.cfg.timeout_s)
            except (requests.ConnectionError, requests.Timeout) as e:
                raise RetryableError(str(e)) from e
            if resp.status_code == 404:
                return None  # not published (yet), or before the archive starts
            if resp.status_code == 429 or resp.status_code >= 500:
                raise RetryableError(f"HTTP {resp.status_code}")
            if resp.status_code != 200:
                raise RuntimeError(f"elprisetjustnu HTTP {resp.status_code} for {url}")
            body: list[dict[str, Any]] = resp.json()
            return body

        return with_retries(call, max_retries=self.cfg.max_retries)

    @staticmethod
    def parse_day(entries: list[dict[str, Any]]) -> pd.DataFrame:
        """One day's entries -> price table.

        Resolution comes from the spacing of `time_start`, not from `time_end`: the API computes
        `time_end` in wall-clock time, so the last slot before the autumn DST change claims
        75 minutes (02:45+02:00 -> 03:00+01:00). See `time_end_anomalies`.
        """
        df = pd.DataFrame(entries)
        start = pd.to_datetime(df["time_start"], utc=True)
        return pd.DataFrame(
            {
                "timestamp": start,
                # EUR/kWh has 5 decimals, so EUR/MWh is exact to 0.01.
                "price_eur_mwh": (df["EUR_per_kWh"].astype("float64") * 1000).round(2),
                "resolution": infer_row_resolution(pd.DatetimeIndex(start)).to_numpy(),
                "sek_per_kwh": df["SEK_per_kWh"].astype("float64"),
                "eur_per_kwh": df["EUR_per_kWh"].astype("float64"),
                "exr": df["EXR"].astype("float64"),
                "time_start_local": df["time_start"].astype("string"),
                "time_end_local": df["time_end"].astype("string"),
            }
        )

    @staticmethod
    def time_end_anomalies(df: pd.DataFrame) -> list[str]:
        """Rows whose reported end differs from the next row's start (within a day)."""
        end = pd.to_datetime(df["time_end_local"], utc=True)
        nxt = df["timestamp"].shift(-1)
        bad = nxt.notna() & (end != nxt)
        return df.loc[bad, "time_start_local"].astype(str).tolist()

    def fetch_prices(self, zone: str, first: date, last: date, tz: str) -> PricePull:
        frames, missing, anomalies = [], [], []
        day = first
        while day <= last:
            entries = self._get_day(zone, day)
            if entries:
                parsed = self.parse_day(entries)
                anomalies.extend(self.time_end_anomalies(parsed))
                frames.append(parsed)
            else:
                missing.append(day.isoformat())
            day += timedelta(days=1)
        data = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame({c: [] for c in PRICE_COLUMNS})
        )
        data = data.sort_values("timestamp").reset_index(drop=True)
        return PricePull(
            source=self.name,
            zone=zone,
            data=data,
            requested_start=local_day_bounds_utc(first, tz)[0],
            requested_end=local_day_bounds_utc(last, tz)[1],
            query={
                "zone": zone,
                "first_day": first.isoformat(),
                "last_day": last.isoformat(),
                "url_template": self.cfg.url_template,
            },
            endpoint=self.cfg.url_template,
            chunks=[
                {
                    "missing_days": missing,
                    "days_requested": (last - first).days + 1,
                    "time_end_anomalies": anomalies,
                }
            ],
            extra={
                "attribution": "Electricity prices from elprisetjustnu.se",
                "price_basis": "excl. VAT, surcharges and taxes",
            },
        )


class EntsoePriceSource:
    """ENTSO-E day-ahead prices (document type A44) through the existing fetcher."""

    name = "entsoe"

    def __init__(self, cfg: EntsoeConfig, client: Any) -> None:
        from pricefc.ingest.entsoe import EntsoeFetcher

        self.fetcher = EntsoeFetcher(cfg, client)

    def fetch_prices(self, zone: str, first: date, last: date, tz: str) -> PricePull:
        from pricefc.ingest.entsoe import price_job

        start, _ = local_day_bounds_utc(first, tz)
        _, end = local_day_bounds_utc(last, tz)
        job = price_job(zone)
        # The fetcher works on UTC dates; fetch the covering range, then trim to local days.
        pull = self.fetcher.fetch(job, start.date(), end.date() + timedelta(days=1))
        data = pull.data
        if len(data):
            data = data[(data["timestamp"] >= start) & (data["timestamp"] < end)]
        return PricePull(
            self.name,
            zone,
            data.reset_index(drop=True),
            start,
            end,
            job.query,
            "entsoe-py",
            pull.chunks,
        )
