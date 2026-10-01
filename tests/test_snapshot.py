from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from pricefc.ingest.snapshot import (
    SnapshotExistsError,
    data_hash,
    list_snapshots,
    load_latest,
    write_snapshot,
)

T0 = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
T1 = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)


def frame(start: str, values: list[float]) -> pd.DataFrame:
    ts = pd.date_range(start, periods=len(values), freq="h", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "value": values})


def write(root: Path, df: pd.DataFrame, pulled_at: datetime, passed: bool = True):  # type: ignore[no-untyped-def]
    return write_snapshot(
        df,
        raw_root=root,
        source="test",
        dataset="ds",
        key="k1",
        endpoint="http://example",
        query={"a": 1},
        requested_start=df["timestamp"].min(),
        requested_end=df["timestamp"].max(),
        pulled_at=pulled_at,
        validation={"passed": passed},
    )


def test_manifest_contents(tmp_path: Path) -> None:
    df = frame("2026-01-01", [1.0, 2.0, 3.0])
    snap = write(tmp_path, df, T0)
    assert snap.path.name == "pulled_at=2026-09-01T12-00-00Z"
    m = snap.manifest
    assert m["row_count"] == 3
    assert m["data_sha256"] == data_hash(df)
    assert {"git_sha", "library_versions", "content_sha256", "query"} <= m.keys()
    pd.testing.assert_frame_equal(snap.read(), df)


def test_snapshots_are_immutable(tmp_path: Path) -> None:
    df = frame("2026-01-01", [1.0])
    snap = write(tmp_path, df, T0)
    with pytest.raises(SnapshotExistsError):
        write(tmp_path, df, T0)
    with pytest.raises(PermissionError):
        snap.data_file.write_bytes(b"x")


def test_unsafe_key_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        write_snapshot(
            frame("2026-01-01", [1.0]),
            raw_root=tmp_path,
            source="test",
            dataset="ds",
            key="../escape",
            endpoint="",
            query={},
            requested_start=pd.Timestamp(0, tz="UTC"),
            requested_end=pd.Timestamp(0, tz="UTC"),
            pulled_at=T0,
            validation={"passed": True},
        )


def test_load_latest_prefers_newest_pull_and_skips_invalid(tmp_path: Path) -> None:
    write(tmp_path, frame("2026-01-01 00:00", [1.0, 2.0, 3.0]), T0)
    write(tmp_path, frame("2026-01-01 02:00", [30.0, 40.0]), T1)
    write(tmp_path, frame("2026-01-01 00:00", [-9.0]), datetime(2026, 9, 3, tzinfo=UTC), False)

    df, used = load_latest(tmp_path, "test", "ds", "k1")
    assert df["value"].tolist() == [1.0, 2.0, 30.0, 40.0]
    assert len(used) == 2

    as_of, used = load_latest(tmp_path, "test", "ds", "k1", as_of=T0)
    assert as_of["value"].tolist() == [1.0, 2.0, 3.0]
    assert len(list_snapshots(tmp_path, "test", "ds", "k1", only_valid=False)) == 3
