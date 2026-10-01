"""Pandera-based validation of raw time series (spec section 5.4).

Checks: UTC-aware timestamps, strictly increasing and unique, value ranges (hard errors) and
plausibility flags (warnings, e.g. negative prices), null rates, maximum gap length, and
DST-aware expected row counts per local day. Nothing is imputed here.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors, SchemaWarning

from pricefc.timeutils import (
    MTU15_GO_LIVE,
    PT15M,
    PT60M,
    RESOLUTION_STEPS,
    expected_periods,
    local_day_bounds_utc,
)


@dataclass(frozen=True)
class ColumnSpec:
    min: float | None = None
    max: float | None = None
    warn_below: float | None = None
    warn_above: float | None = None
    max_null_frac: float = 0.0


@dataclass(frozen=True)
class SeriesSpec:
    """Validation rules for one raw dataset.

    `step`: a key of RESOLUTION_STEPS for fixed-step series, "per_row" when the table has a
    `resolution` column (prices across the 15-minute switch), or None for irregular data.
    `columns`: rules for named columns; `default_column` applies to other numeric columns.
    """

    name: str
    step: str | None
    columns: dict[str, ColumnSpec] = field(default_factory=dict)
    default_column: ColumnSpec | None = None
    max_gap: pd.Timedelta | None = None
    check_day_counts: bool = True
    unique_timestamps: bool = True


@dataclass
class ValidationReport:
    dataset: str
    passed: bool
    errors: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> dict[str, Any]:
        """Compact form stored in the snapshot manifest."""
        return {
            "passed": self.passed,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "errors": self.errors[:20],
            "warnings": self.warnings[:20],
            "stats": self.stats,
        }


class ValidationFailedError(RuntimeError):
    def __init__(self, report: ValidationReport) -> None:
        super().__init__(f"validation failed for {report.dataset}: {report.errors[:5]}")
        self.report = report


def _is_utc(s: pd.Series) -> bool:
    return isinstance(s.dtype, pd.DatetimeTZDtype) and str(s.dtype.tz) == "UTC"


def _column_checks(name: str, spec: ColumnSpec) -> list[pa.Check]:
    checks = []
    if spec.min is not None:
        checks.append(pa.Check.ge(spec.min, ignore_na=True))
    if spec.max is not None:
        checks.append(pa.Check.le(spec.max, ignore_na=True))
    if spec.warn_below is not None:
        checks.append(
            pa.Check.ge(spec.warn_below, ignore_na=True, raise_warning=True, name=f"{name}_low")
        )
    if spec.warn_above is not None:
        checks.append(
            pa.Check.le(spec.warn_above, ignore_na=True, raise_warning=True, name=f"{name}_high")
        )
    frac = spec.max_null_frac
    checks.append(
        pa.Check(
            lambda s, f=frac: bool(s.isna().mean() <= f), name=f"null_frac<={frac}", ignore_na=False
        )
    )
    return checks


def build_schema(spec: SeriesSpec, df: pd.DataFrame) -> pa.DataFrameSchema:
    columns: dict[str, pa.Column] = {
        "timestamp": pa.Column(
            checks=[
                pa.Check(_is_utc, name="tz_aware_utc"),
                pa.Check(lambda s: bool(s.is_monotonic_increasing), name="monotonic"),
            ],
            unique=spec.unique_timestamps,
            nullable=False,
        )
    }
    for col in df.columns:
        if col == "timestamp":
            continue
        col_spec = spec.columns.get(col)
        if (
            col_spec is None
            and spec.default_column is not None
            and pd.api.types.is_numeric_dtype(df[col])
        ):
            col_spec = spec.default_column
        if col_spec is not None:
            columns[col] = pa.Column(
                checks=[
                    pa.Check(lambda s: bool(pd.api.types.is_numeric_dtype(s)), name="numeric"),
                    *_column_checks(col, col_spec),
                ],
                nullable=True,
            )
    for col, col_spec in spec.columns.items():
        if col not in columns:
            columns[col] = pa.Column(checks=_column_checks(col, col_spec), nullable=True)
    return pa.DataFrameSchema(columns, strict=False)


def _gap_errors(ts: pd.Series, max_gap: pd.Timedelta) -> list[dict[str, Any]]:
    diffs = ts.diff()
    bad = diffs[diffs > max_gap]
    return [
        {"check": "max_gap", "gap_end": ts[i].isoformat(), "gap": str(bad[i])} for i in bad.index
    ]


def _day_count_errors(
    df: pd.DataFrame,
    spec: SeriesSpec,
    tz: str,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
) -> list[dict[str, Any]]:
    """Every local day fully inside the requested range must have the expected row count."""
    errors: list[dict[str, Any]] = []
    local_days = df["timestamp"].dt.tz_convert(tz).dt.date
    counts = local_days.value_counts()
    first = requested_start.tz_convert(tz).date()
    last = requested_end.tz_convert(tz).date()
    day = first
    while day <= last:
        start, end = local_day_bounds_utc(day, tz)
        if start >= requested_start and end <= requested_end:
            mask = local_days == day
            if spec.step == "per_row":
                res_values = df.loc[mask, "resolution"].dropna().unique().tolist()
                if len(res_values) > 1:
                    errors.append({"check": "day_resolution", "day": str(day), "got": res_values})
                    day += timedelta(days=1)
                    continue
                # A day with no rows: expect the market's resolution for that delivery day.
                default = PT15M if day >= MTU15_GO_LIVE else PT60M
                res = res_values[0] if res_values else default
            else:
                res = spec.step
            got = int(counts.get(day, 0))
            want = expected_periods(day, tz, res) if res in RESOLUTION_STEPS else None
            if want is None or got != want:
                errors.append({"check": "day_count", "day": str(day), "expected": want, "got": got})
        day += timedelta(days=1)
    return errors


def _failure_cases(err: SchemaErrors) -> list[dict[str, Any]]:
    fc = err.failure_cases
    out = []
    for check, grp in fc.groupby("check", dropna=False):
        cols = grp["column"].dropna().unique().tolist()
        examples = grp["failure_case"].head(5).astype(str).tolist()
        out.append({"check": str(check), "columns": cols, "n": len(grp), "examples": examples})
    return out


def validate_series(
    df: pd.DataFrame,
    spec: SeriesSpec,
    *,
    tz: str,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
) -> ValidationReport:
    errors: list[dict[str, Any]] = []
    warns: list[dict[str, Any]] = []
    if "timestamp" not in df.columns:
        errors.append({"check": "has_timestamp_column"})
    else:
        schema = build_schema(spec, df)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", SchemaWarning)
            try:
                schema.validate(df, lazy=True)
            except SchemaErrors as e:
                errors.extend(_failure_cases(e))
        for w in caught:
            if issubclass(w.category, SchemaWarning):
                warns.append({"check": "warning", "message": str(w.message)[:500]})

        structural_ok = not errors and len(df) > 0
        if structural_ok and spec.max_gap is not None:
            errors.extend(_gap_errors(df["timestamp"], spec.max_gap))
        if structural_ok and spec.check_day_counts and spec.step is not None:
            errors.extend(_day_count_errors(df, spec, tz, requested_start, requested_end))
        if len(df) == 0:
            errors.append({"check": "non_empty"})

    stats: dict[str, Any] = {"rows": len(df)}
    if "timestamp" in df.columns and len(df):
        stats["start"] = str(df["timestamp"].min())
        stats["end"] = str(df["timestamp"].max())
    nulls = {str(c): int(n) for c, n in df.isna().sum().items() if n > 0}
    if nulls:
        stats["null_counts"] = nulls
    if "resolution" in df.columns:
        stats["resolution_counts"] = {
            str(k): int(v) for k, v in df["resolution"].value_counts(dropna=False).items()
        }
    return ValidationReport(spec.name, not errors, errors, warns, stats)
