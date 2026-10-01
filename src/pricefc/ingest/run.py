"""Ingestion runs: fetch -> validate -> immutable snapshot -> MLflow lineage."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal

import structlog

from pricefc.config import BaseConfig, FeaturesConfig, IngestConfig, Location
from pricefc.ingest import entsoe as es
from pricefc.ingest import openmeteo as om
from pricefc.ingest.prices import PriceSource, month_ranges
from pricefc.ingest.snapshot import Snapshot, fresh_pulled_at, list_snapshots, write_snapshot
from pricefc.validate.schemas import ValidationReport, validate_series
from pricefc.validate.specs import entsoe_spec, weather_spec

log = structlog.get_logger(__name__)
WeatherEndpoint = Literal["historical_forecast", "previous_runs", "forecast"]


@dataclass
class IngestResult:
    snapshot: Snapshot
    report: ValidationReport


def now_utc() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def locations_for(features: FeaturesConfig, zones: list[str]) -> list[Location]:
    out = []
    for z in zones:
        if z not in features.weather.locations:
            raise KeyError(f"no weather locations configured for zone {z}")
        out.extend(features.weather.locations[z])
    return out


def ingest_weather(
    base: BaseConfig,
    ingest: IngestConfig,
    locations: list[Location],
    endpoint: WeatherEndpoint,
    start: date | None = None,
    end: date | None = None,
    client: om.OpenMeteoClient | None = None,
) -> list[IngestResult]:
    client = client or om.OpenMeteoClient(ingest.open_meteo)
    ep_cfg = ingest.open_meteo.endpoints[endpoint]
    results = []
    for loc in locations:
        if endpoint == "forecast":
            pull = client.fetch_forecast(loc)
        else:
            s = start or ep_cfg.start
            if s is None:
                raise ValueError(f"no start date for {endpoint}")
            e = end or now_utc().date() - timedelta(days=1)  # last complete UTC day
            pull = client.fetch_range(endpoint, loc, s, e)
        pulled_at = fresh_pulled_at(base.paths.raw, om.SOURCE, pull.dataset, loc.name)
        spec = weather_spec(
            pull.dataset, list(pull.data.columns), max_null_frac=ep_cfg.max_null_frac
        )
        report = validate_series(
            pull.data,
            spec,
            tz=base.timezone,
            requested_start=pull.requested_start,
            requested_end=pull.requested_end,
        )
        snap = write_snapshot(
            pull.data,
            raw_root=base.paths.raw,
            source=om.SOURCE,
            dataset=pull.dataset,
            key=loc.name,
            endpoint=pull.query["url"],
            query=pull.query,
            requested_start=pull.requested_start,
            requested_end=pull.requested_end,
            pulled_at=pulled_at,
            validation=report.summary(),
            extra={
                "chunks": pull.chunks,
                "location": loc.model_dump(),
                "weather_source": om.weather_source_tag(endpoint, ingest.open_meteo.model),
                "attribution": "Weather data by Open-Meteo.com (CC BY 4.0)",
            },
        )
        log.info(
            "snapshot_written",
            dataset=pull.dataset,
            key=loc.name,
            rows=len(pull.data),
            valid=report.passed,
            path=str(snap.path),
        )
        results.append(IngestResult(snap, report))
    return results


def make_entsoe_client() -> es.EntsoeClient:
    from entsoe.entsoe import EntsoePandasClient

    token = os.environ.get("ENTSOE_API_TOKEN")
    if not token:
        raise RuntimeError("ENTSOE_API_TOKEN is not set (see .env.example)")
    client: es.EntsoeClient = EntsoePandasClient(api_key=token)
    return client


def ingest_entsoe(
    base: BaseConfig,
    ingest: IngestConfig,
    jobs: list[es.EntsoeJob],
    start: date | None = None,
    end: date | None = None,
    client: es.EntsoeClient | None = None,
) -> list[IngestResult]:
    fetcher = es.EntsoeFetcher(ingest.entsoe, client or make_entsoe_client())
    s = start or ingest.entsoe.history_start
    e = end or now_utc().date()
    results = []
    for job in jobs:
        pull = fetcher.fetch(job, s, e)
        pulled_at = fresh_pulled_at(base.paths.raw, es.SOURCE, job.dataset, job.key)
        report = validate_series(
            pull.data,
            entsoe_spec(job.dataset),
            tz=base.timezone,
            requested_start=pull.requested_start,
            requested_end=pull.requested_end,
        )
        snap = write_snapshot(
            pull.data,
            raw_root=base.paths.raw,
            source=es.SOURCE,
            dataset=job.dataset,
            key=job.key,
            endpoint="entsoe-py",
            query=job.query,
            requested_start=pull.requested_start,
            requested_end=pull.requested_end,
            pulled_at=pulled_at,
            validation=report.summary(),
            extra={
                "chunks": pull.chunks,
                "known_limitations": ["forecast series have no issue-time vintages"],
            },
        )
        log.info(
            "snapshot_written",
            dataset=job.dataset,
            key=job.key,
            rows=len(pull.data),
            valid=report.passed,
            path=str(snap.path),
        )
        results.append(IngestResult(snap, report))
    return results


def log_ingest_run(
    base: BaseConfig,
    results: list[IngestResult],
    *,
    zones: list[str],
    label: str,
    weather_source: str = "none",
) -> str:
    """Record an ingest batch in MLflow: manifests, validation reports, row counts."""
    import mlflow

    from pricefc.tracking.mlflow_utils import base_tags, setup_tracking, start_run

    setup_tracking(base)
    tags = base_tags(
        base, zone=",".join(zones), pipeline_stage="ingest", weather_source=weather_source
    )
    with start_run("ingest", tags, base, run_name=label) as run:
        n_failed = sum(not r.report.passed for r in results)
        mlflow.log_metrics(
            {
                "snapshots": len(results),
                "validation_failures": n_failed,
                "rows": sum(r.snapshot.manifest["row_count"] for r in results),
            }
        )
        index: list[dict[str, Any]] = []
        for r in results:
            m = r.snapshot.manifest
            name = f"{m['source']}__{m['dataset']}__{m['key']}"
            mlflow.log_dict(m, f"manifests/{name}.json")
            mlflow.log_dict(r.report.to_dict(), f"validation/{name}.json")
            index.append(
                {
                    "path": str(r.snapshot.path),
                    "valid": r.report.passed,
                    "rows": m["row_count"],
                    "data_sha256": m["data_sha256"],
                }
            )
        mlflow.log_text(json.dumps(index, indent=2), "snapshots.json")
        return str(run.info.run_id)


def _month_done(base: BaseConfig, source: str, zone: str, first: date, last: date) -> bool:
    for snap in list_snapshots(base.paths.raw, source, "day_ahead_prices", zone):
        q = snap.manifest["query"]
        if (
            q.get("first_day", "") <= first.isoformat()
            and q.get("last_day", "") >= last.isoformat()
        ):
            return True
    return False


def ingest_prices(
    base: BaseConfig,
    source: PriceSource,
    zones: list[str],
    first: date,
    last: date,
    *,
    force: bool = False,
) -> list[IngestResult]:
    """Pull prices per zone and calendar month; complete months already stored are skipped.

    Resumable: an interrupted backfill restarts at the first month without a valid snapshot.
    The month containing `last` is re-pulled on every run unless `last` is its final day.
    """
    results = []
    for zone in zones:
        for m_first, m_last in month_ranges(first, last):
            if not force and _month_done(base, source.name, zone, m_first, m_last):
                continue
            pull = source.fetch_prices(zone, m_first, m_last, base.timezone)
            pulled_at = fresh_pulled_at(base.paths.raw, source.name, "day_ahead_prices", zone)
            report = validate_series(
                pull.data,
                entsoe_spec("day_ahead_prices"),
                tz=base.timezone,
                requested_start=pull.requested_start,
                requested_end=pull.requested_end,
            )
            snap = write_snapshot(
                pull.data,
                raw_root=base.paths.raw,
                source=source.name,
                dataset="day_ahead_prices",
                key=zone,
                endpoint=pull.endpoint,
                query=pull.query,
                requested_start=pull.requested_start,
                requested_end=pull.requested_end,
                pulled_at=pulled_at,
                validation=report.summary(),
                extra={"chunks": pull.chunks, "price_source": source.name, **pull.extra},
            )
            log.info(
                "snapshot_written",
                source=source.name,
                zone=zone,
                month=m_first.strftime("%Y-%m"),
                rows=len(pull.data),
                valid=report.passed,
            )
            results.append(IngestResult(snap, report))
    return results
