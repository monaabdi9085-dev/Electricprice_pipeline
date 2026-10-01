"""Validation rules per raw dataset."""

from __future__ import annotations

import pandas as pd

from pricefc.validate.schemas import ColumnSpec, SeriesSpec

# Harmonised SDAC limits have been -500 and +4000 EUR/MWh; the hard bounds are deliberately
# wider so a limit change is flagged by the warning, not rejected.
PRICE = ColumnSpec(min=-1000, max=10000, warn_below=0.0, warn_above=1000.0)
POWER_MW = ColumnSpec(min=0, max=100_000, max_null_frac=0.05)
EXCHANGE_MW = ColumnSpec(min=-100_000, max=100_000, max_null_frac=0.05)

# Weather variables by prefix (Previous Runs columns carry a `_previous_dayN` suffix).
WEATHER_RANGES: dict[str, ColumnSpec] = {
    "temperature_2m": ColumnSpec(min=-60, max=50),
    "wind_speed": ColumnSpec(min=0, max=80),
    "wind_direction": ColumnSpec(min=0, max=360),
    "cloud_cover": ColumnSpec(min=0, max=100),
    # Model output can dip slightly below zero at night (seen: -1 W/m2); flag, keep as is.
    "shortwave_radiation": ColumnSpec(min=-5, max=1500, warn_below=0),
    "precipitation": ColumnSpec(min=0, max=300),
    "snowfall": ColumnSpec(min=0, max=200),
}


def entsoe_spec(dataset: str) -> SeriesSpec:
    if dataset == "day_ahead_prices":
        return SeriesSpec(
            dataset, step="per_row", columns={"price_eur_mwh": PRICE}, max_gap=pd.Timedelta(hours=1)
        )
    if dataset in ("load_forecast", "wind_solar_forecast", "generation"):
        return SeriesSpec(
            dataset,
            step="per_row",
            default_column=POWER_MW,
            max_gap=pd.Timedelta(hours=6),
            check_day_counts=False,
        )
    if dataset == "crossborder_flows":
        return SeriesSpec(
            dataset,
            step="per_row",
            default_column=POWER_MW,
            max_gap=pd.Timedelta(hours=6),
            check_day_counts=False,
        )
    if dataset == "scheduled_exchanges":
        return SeriesSpec(
            dataset,
            step="per_row",
            default_column=EXCHANGE_MW,
            max_gap=pd.Timedelta(hours=6),
            check_day_counts=False,
        )
    if dataset == "hydro_reservoirs":
        return SeriesSpec(
            dataset, step=None, default_column=ColumnSpec(min=0), max_gap=pd.Timedelta(days=15)
        )
    if dataset == "generation_unavailability":
        return SeriesSpec(dataset, step=None, unique_timestamps=False)
    raise KeyError(dataset)


def weather_spec(dataset: str, columns: list[str], *, max_null_frac: float = 0.0) -> SeriesSpec:
    specs = {}
    for col in columns:
        for prefix, rng in WEATHER_RANGES.items():
            if col.startswith(prefix):
                specs[col] = ColumnSpec(
                    min=rng.min,
                    max=rng.max,
                    warn_below=rng.warn_below,
                    max_null_frac=max_null_frac,
                )
                break
        else:
            if col != "timestamp":
                raise KeyError(f"no validation range for weather column {col!r}")
    return SeriesSpec(dataset, step="PT60M", columns=specs, max_gap=pd.Timedelta(hours=1))
