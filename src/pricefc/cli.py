"""Command-line entrypoint: `python -m pricefc ...`."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import typer
import yaml
from dotenv import load_dotenv

from pricefc.config import BaseConfig, config_hash, load_config

if TYPE_CHECKING:
    from pricefc.ingest.run import IngestResult

app = typer.Typer(no_args_is_help=True, add_completion=False)

CONFIG_OPT = typer.Option(Path("configs/base.yaml"), "--config", "-c", help="Base config file.")


@app.callback()
def _main() -> None:
    load_dotenv()


@app.command()
def show_config(config: Path = CONFIG_OPT) -> None:
    """Print the resolved config and its hash."""
    cfg = load_config(config)
    typer.echo(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
    typer.echo(f"config_hash: {config_hash(cfg)}")


@app.command()
def init_tracking(
    config: Path = CONFIG_OPT,
    tracking_uri: str | None = typer.Option(None, help="Overrides config mlflow.tracking_uri."),
) -> None:
    """Create the standard MLflow experiments."""
    from pricefc.tracking.mlflow_utils import EXPERIMENTS, setup_tracking

    setup_tracking(load_config(config), tracking_uri)
    typer.echo(f"experiments ready: {', '.join(EXPERIMENTS)}")


ingest_app = typer.Typer(no_args_is_help=True, help="Pull raw data into immutable snapshots.")
app.add_typer(ingest_app, name="ingest")

INGEST_OPT = typer.Option(Path("configs/ingest.yaml"), "--ingest-config")
FEATURES_OPT = typer.Option(Path("configs/features.yaml"), "--features-config")
DATASET_OPT = typer.Option(None, "--dataset", "-d", help="Repeatable.")
PRICE_AREA_OPT = typer.Option(
    None, "--price-area", help="Pull day-ahead prices for these areas only (e.g. NO1)."
)
ZONE_OPT = typer.Option(None, "--zone", "-z", help="Repeatable. Default: zones in base config.")
START_OPT = typer.Option(None, formats=["%Y-%m-%d"], help="UTC start date.")
END_OPT = typer.Option(None, formats=["%Y-%m-%d"], help="UTC end date.")


def _date(d: datetime | None) -> date | None:
    return d.date() if d else None


def _finish(
    results: list[IngestResult],
    label: str,
    zones: list[str],
    base: BaseConfig,
    weather_source: str,
) -> None:
    from pricefc.ingest.run import log_ingest_run

    run_id = log_ingest_run(base, results, zones=zones, label=label, weather_source=weather_source)
    failed = [r for r in results if not r.report.passed]
    for r in results:
        status = "ok" if r.report.passed else "FAILED"
        typer.echo(f"[{status}] {r.snapshot.path} rows={r.snapshot.manifest['row_count']}")
    typer.echo(f"mlflow ingest run: {run_id}")
    if failed:
        typer.echo(f"{len(failed)} snapshot(s) failed validation; see MLflow run.", err=True)
        raise typer.Exit(1)


@ingest_app.command("weather")
def ingest_weather_cmd(
    endpoint: str = typer.Argument(..., help="historical_forecast | previous_runs | forecast"),
    zone: list[str] | None = ZONE_OPT,
    start: datetime | None = START_OPT,
    end: datetime | None = END_OPT,
    config: Path = CONFIG_OPT,
    ingest_config: Path = INGEST_OPT,
    features_config: Path = FEATURES_OPT,
) -> None:
    """Pull Open-Meteo weather for every configured location of the zone(s)."""
    from pricefc.config import load_features_config, load_ingest_config
    from pricefc.ingest.openmeteo import ENDPOINTS, weather_source_tag
    from pricefc.ingest.run import WeatherEndpoint, ingest_weather, locations_for

    if endpoint not in ENDPOINTS:
        raise typer.BadParameter(f"endpoint must be one of {ENDPOINTS}")
    base = load_config(config)
    ing = load_ingest_config(ingest_config)
    zones = zone or list(base.zones)
    locs = locations_for(load_features_config(features_config), zones)
    ep = cast("WeatherEndpoint", endpoint)
    results = ingest_weather(base, ing, locs, ep, _date(start), _date(end))
    _finish(
        results,
        f"weather-{endpoint}",
        zones,
        base,
        weather_source_tag(endpoint, ing.open_meteo.model),
    )


@ingest_app.command("entsoe")
def ingest_entsoe_cmd(
    zone: list[str] | None = ZONE_OPT,
    dataset: list[str] | None = DATASET_OPT,
    price_area: list[str] | None = PRICE_AREA_OPT,
    start: datetime | None = START_OPT,
    end: datetime | None = END_OPT,
    config: Path = CONFIG_OPT,
    ingest_config: Path = INGEST_OPT,
) -> None:
    """Pull ENTSO-E datasets for the zone(s). Requires ENTSOE_API_TOKEN."""
    from pricefc.config import load_ingest_config
    from pricefc.ingest.entsoe import jobs_for_zone, price_job
    from pricefc.ingest.run import ingest_entsoe

    base = load_config(config)
    ing = load_ingest_config(ingest_config)
    zones = zone or list(base.zones)
    if price_area:
        jobs = [price_job(a) for a in price_area]
    else:
        jobs = [j for z in zones for j in jobs_for_zone(z, ing.entsoe)]
        if dataset:
            jobs = [j for j in jobs if j.dataset in dataset]
    # Neighbouring zones share exchange series and the hydro area; pull each once.
    jobs = list({(j.dataset, j.key): j for j in jobs}.values())
    results = ingest_entsoe(base, ing, jobs, _date(start), _date(end))
    _finish(results, "entsoe", zones, base, "none")


@ingest_app.command("prices")
def ingest_prices_cmd(
    zone: list[str] | None = ZONE_OPT,
    start: datetime | None = START_OPT,
    end: datetime | None = END_OPT,
    source: str | None = typer.Option(None, help="Override prices.source from ingest config."),
    force: bool = typer.Option(False, help="Re-pull months that already have a snapshot."),
    config: Path = CONFIG_OPT,
    ingest_config: Path = INGEST_OPT,
) -> None:
    """Pull day-ahead prices (local delivery days) per zone and month. Resumable."""
    from zoneinfo import ZoneInfo

    from pricefc.config import load_ingest_config
    from pricefc.ingest.prices import ElprisSource, EntsoePriceSource, PriceSource
    from pricefc.ingest.run import ingest_prices, make_entsoe_client

    base = load_config(config)
    ing = load_ingest_config(ingest_config)
    zones = zone or list(base.zones)
    name = source or ing.prices.source
    src: PriceSource
    if name == "elprisetjustnu":
        src = ElprisSource(ing.elprisetjustnu)
        default_first = ing.elprisetjustnu.start
    elif name == "entsoe":
        src = EntsoePriceSource(ing.entsoe, make_entsoe_client())
        default_first = ing.entsoe.history_start
    else:
        raise typer.BadParameter(f"unknown price source {name!r}")
    today = datetime.now(ZoneInfo(base.timezone)).date()
    first = _date(start) or default_first
    last = _date(end) or today
    results = ingest_prices(base, src, zones, first, last, force=force)
    if not results:
        typer.echo("nothing to do: all months already stored")
        return
    _finish(results, f"prices-{name}", zones, base, "none")


@app.command()
def snapshots(config: Path = CONFIG_OPT) -> None:
    """List raw snapshots with row counts and validation status."""
    import json

    root = load_config(config).paths.raw
    for manifest in sorted(root.glob("*/*/*/pulled_at=*/manifest.json")):
        m = json.loads(manifest.read_text())
        status = "ok" if m["validation"]["passed"] else "FAILED"
        typer.echo(
            f"[{status}] {m['source']}/{m['dataset']}/{m['key']} "
            f"pulled={m['pulled_at']} rows={m['row_count']} "
            f"{m['data_start']} .. {m['data_end']}"
        )


if __name__ == "__main__":
    app()
