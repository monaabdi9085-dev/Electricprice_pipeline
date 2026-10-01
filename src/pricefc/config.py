"""Configuration loading, validation and hashing."""

from __future__ import annotations

import hashlib
import json
from datetime import date, time
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

Zone = Literal["SE1", "SE2", "SE3", "SE4"]
Resolution = Literal["hourly", "PT15M"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OriginConfig(_Strict):
    local_time: time


class PathsConfig(_Strict):
    data_root: Path
    raw: Path
    datasets: Path


class MlflowConfig(_Strict):
    tracking_uri: str
    artifact_root: str


class BaseConfig(_Strict):
    zones: list[Zone] = Field(min_length=1)
    resolution: Resolution
    timezone: str
    origin: OriginConfig
    quantiles: list[float] = Field(min_length=1)
    history_start: date
    seed: int
    paths: PathsConfig
    mlflow: MlflowConfig
    compute_env: str

    @field_validator("quantiles")
    @classmethod
    def _check_quantiles(cls, v: list[float]) -> list[float]:
        if any(not 0.0 < q < 1.0 for q in v):
            raise ValueError("quantiles must lie strictly between 0 and 1")
        if sorted(set(v)) != v:
            raise ValueError("quantiles must be unique and sorted ascending")
        if 0.5 not in v:
            raise ValueError("quantiles must include 0.5 (the point forecast)")
        return v

    @field_validator("zones")
    @classmethod
    def _unique_zones(cls, v: list[Zone]) -> list[Zone]:
        if len(set(v)) != len(v):
            raise ValueError("zones must be unique")
        return v


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def load_config(path: Path, overrides: dict[str, Any] | None = None) -> BaseConfig:
    raw = load_yaml(path)
    if overrides:
        raw = _deep_merge(raw, overrides)
    return BaseConfig.model_validate(raw)


def config_hash(config: BaseModel) -> str:
    """Stable short hash of a resolved config (logged as the `config_hash` tag)."""
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


# --- ingestion -------------------------------------------------------------------------------


class OpenMeteoEndpoint(_Strict):
    url: str
    start: date | None = None
    lead_days: list[int] = Field(default_factory=list)
    forecast_days: int | None = None
    max_null_frac: float = Field(default=0.0, ge=0, le=1)


class OpenMeteoConfig(_Strict):
    model: str
    variables: list[str] = Field(min_length=1)
    wind_speed_unit: Literal["ms", "kmh", "mph", "kn"]
    min_interval_s: float = Field(ge=0)
    max_retries: int = Field(ge=0)
    timeout_s: float = Field(gt=0)
    endpoints: dict[Literal["historical_forecast", "previous_runs", "forecast"], OpenMeteoEndpoint]


EntsoeDataset = Literal[
    "day_ahead_prices",
    "load_forecast",
    "wind_solar_forecast",
    "generation",
    "crossborder_flows",
    "scheduled_exchanges",
    "generation_unavailability",
    "hydro_reservoirs",
]


class EntsoeConfig(_Strict):
    min_interval_s: float = Field(ge=0)
    max_retries: int = Field(ge=0)
    chunk_days: int = Field(gt=0, le=366)
    history_start: date
    datasets: list[EntsoeDataset]
    neighbour_price_areas: list[str]
    hydro_reservoir_area: str


class PricesConfig(_Strict):
    source: Literal["elprisetjustnu", "entsoe"]


class ElprisConfig(_Strict):
    url_template: str
    start: date
    min_interval_s: float = Field(ge=0)
    max_retries: int = Field(ge=0)
    timeout_s: float = Field(gt=0)
    user_agent: str


class IngestConfig(_Strict):
    prices: PricesConfig
    elprisetjustnu: ElprisConfig
    open_meteo: OpenMeteoConfig
    entsoe: EntsoeConfig


def load_ingest_config(path: Path) -> IngestConfig:
    return IngestConfig.model_validate(load_yaml(path))


# --- features --------------------------------------------------------------------------------


class Location(_Strict):
    name: str = Field(pattern=r"^[a-z0-9_]+$")
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    role: Literal["demand", "wind", "solar", "neighbour_wind"]


class WeatherFeatures(_Strict):
    locations: dict[Zone, list[Location]]

    @field_validator("locations")
    @classmethod
    def _unique_names(cls, v: dict[Zone, list[Location]]) -> dict[Zone, list[Location]]:
        names = [loc.name for locs in v.values() for loc in locs]
        if len(names) != len(set(names)):
            raise ValueError("location names must be unique across zones")
        return v


class FeaturesConfig(_Strict):
    weather: WeatherFeatures


def load_features_config(path: Path) -> FeaturesConfig:
    return FeaturesConfig.model_validate(load_yaml(path))
