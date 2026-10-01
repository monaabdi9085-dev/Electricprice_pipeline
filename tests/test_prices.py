import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pricefc.config import load_config, load_ingest_config
from pricefc.ingest.prices import ElprisSource, month_ranges
from pricefc.ingest.run import ingest_prices
from pricefc.ingest.throttle import RateLimiter
from pricefc.timeutils import PT15M, PT60M
from pricefc.validate.schemas import validate_series
from pricefc.validate.specs import entsoe_spec

FIX = Path(__file__).parent / "fixtures" / "elprisetjustnu"
CFG = load_ingest_config(Path("configs/ingest.yaml")).elprisetjustnu
TZ = "Europe/Stockholm"


class FakeResponse:
    def __init__(self, status: int, body: Any = None) -> None:
        self.status_code = status
        self._body = body

    def json(self) -> Any:
        return self._body


class FakeSession:
    """Serves recorded fixtures by URL; anything else is a 404."""

    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.urls: list[str] = []

    def get(self, url: str, timeout: float) -> FakeResponse:
        self.urls.append(url)
        year, rest = url.rsplit("/", 2)[-2:]
        path = FIX / f"{year}-{rest}"
        if not path.exists():
            return FakeResponse(404)
        return FakeResponse(200, json.loads(path.read_text()))


def source() -> tuple[ElprisSource, FakeSession]:
    session = FakeSession()
    return ElprisSource(CFG, session=session, limiter=RateLimiter(0)), session  # type: ignore[arg-type]


def test_month_ranges() -> None:
    assert month_ranges(date(2025, 11, 20), date(2026, 1, 3)) == [
        (date(2025, 11, 20), date(2025, 11, 30)),
        (date(2025, 12, 1), date(2025, 12, 31)),
        (date(2026, 1, 1), date(2026, 1, 3)),
    ]


def test_url_and_user_agent() -> None:
    src, session = source()
    assert src.url("SE3", date(2025, 3, 4)).endswith("/prices/2025/03-04_SE3.json")
    assert session.headers["User-Agent"] == CFG.user_agent


@pytest.mark.parametrize(
    ("day", "rows", "res"),
    [
        ("2025-09-30", 24, PT60M),
        ("2025-10-01", 96, PT15M),
        ("2025-10-26", 100, PT15M),  # 25-hour day
        ("2026-03-29", 92, PT15M),  # 23-hour day
        ("2024-03-31", 23, PT60M),  # 23-hour day, hourly era
    ],
)
def test_parse_recorded_days(day: str, rows: int, res: str) -> None:
    entries = json.loads((FIX / f"{day}_SE3.json").read_text())
    df = ElprisSource.parse_day(entries)
    assert len(df) == rows
    assert (df["resolution"] == res).all()
    assert str(df["timestamp"].dt.tz) == "UTC" and df["timestamp"].is_unique
    assert df["timestamp"].iloc[0].tz_convert(TZ).strftime("%Y-%m-%d %H:%M") == f"{day} 00:00"
    assert df["price_eur_mwh"].iloc[0] == pytest.approx(entries[0]["EUR_per_kWh"] * 1000)


def test_fetch_across_mtu_switch_validates() -> None:
    src, _ = source()
    pull = src.fetch_prices("SE3", date(2025, 9, 30), date(2025, 10, 1), TZ)
    assert pull.chunks[0]["missing_days"] == []
    report = validate_series(
        pull.data,
        entsoe_spec("day_ahead_prices"),
        tz=TZ,
        requested_start=pull.requested_start,
        requested_end=pull.requested_end,
    )
    assert report.passed, report.errors
    assert report.stats["resolution_counts"] == {PT60M: 24, PT15M: 96}


def test_missing_day_is_recorded_and_fails_validation() -> None:
    src, _ = source()
    pull = src.fetch_prices("SE3", date(2025, 9, 29), date(2025, 9, 30), TZ)
    assert pull.chunks[0]["missing_days"] == ["2025-09-29"]
    report = validate_series(
        pull.data,
        entsoe_spec("day_ahead_prices"),
        tz=TZ,
        requested_start=pull.requested_start,
        requested_end=pull.requested_end,
    )
    assert not report.passed
    assert {"check": "day_count", "day": "2025-09-29", "expected": 24, "got": 0} in report.errors


def test_ingest_prices_is_resumable(tmp_path: Path) -> None:
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
    src, session = source()
    first = ingest_prices(base, src, ["SE3"], date(2025, 9, 30), date(2025, 10, 1))
    assert [r.report.passed for r in first] == [True, True]
    assert first[0].snapshot.manifest["price_source"] == "elprisetjustnu"
    assert "elprisetjustnu.se" in first[0].snapshot.manifest["attribution"]
    n_calls = len(session.urls)
    again = ingest_prices(base, src, ["SE3"], date(2025, 9, 30), date(2025, 10, 1))
    assert again == [] and len(session.urls) == n_calls
    forced = ingest_prices(base, src, ["SE3"], date(2025, 9, 30), date(2025, 10, 1), force=True)
    assert len(forced) == 2


def test_prices_are_in_eur_per_mwh() -> None:
    src, _ = source()
    df = src.fetch_prices("SE3", date(2025, 10, 26), date(2025, 10, 26), TZ).data
    assert df["price_eur_mwh"].equals((df["eur_per_kwh"] * 1000).round(2))
    assert isinstance(df["timestamp"].dtype, pd.DatetimeTZDtype)


def test_dst_time_end_anomaly_is_recorded_not_trusted() -> None:
    src, _ = source()
    pull = src.fetch_prices("SE3", date(2025, 10, 26), date(2025, 10, 26), TZ)
    assert pull.chunks[0]["time_end_anomalies"] == ["2025-10-26T02:45:00+02:00"]
    assert (pull.data["resolution"] == PT15M).all()
