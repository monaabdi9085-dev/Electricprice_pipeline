"""Immutable raw snapshots (spec section 5.3).

Layout: {raw_root}/{source}/{dataset}/{key}/pulled_at=YYYY-MM-DDTHH-MM-SSZ/
            part-0.parquet
            manifest.json

A snapshot is written once and never overwritten. Readers combine snapshots by taking,
for every timestamp, the value from the most recent pull.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pricefc.lineage import git_info

PULLED_AT_FMT = "%Y-%m-%dT%H-%M-%SZ"
_SAFE = re.compile(r"^[A-Za-z0-9_.\-]+$")
_LIBS = ("pricefc", "pandas", "pyarrow", "numpy", "requests", "entsoe-py", "pandera")


class SnapshotExistsError(FileExistsError):
    """Raised when a snapshot directory already exists (snapshots are immutable)."""


@dataclass(frozen=True)
class Snapshot:
    path: Path
    manifest: dict[str, Any]

    @property
    def data_file(self) -> Path:
        return self.path / "part-0.parquet"

    @property
    def pulled_at(self) -> datetime:
        return datetime.fromisoformat(self.manifest["pulled_at"])

    @property
    def validation_passed(self) -> bool:
        return bool(self.manifest.get("validation", {}).get("passed", False))

    def read(self) -> pd.DataFrame:
        return pd.read_parquet(self.data_file)


def library_versions() -> dict[str, str]:
    out = {}
    for lib in _LIBS:
        try:
            out[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            continue
    return out


def data_hash(df: pd.DataFrame) -> str:
    """Hash of the table contents (row order, values and column names), format-independent."""
    h = hashlib.sha256()
    h.update(json.dumps([str(c) for c in df.columns]).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()


def snapshot_dir(raw_root: Path, source: str, dataset: str, key: str, pulled_at: datetime) -> Path:
    for part in (source, dataset, key):
        if not _SAFE.match(part):
            raise ValueError(f"unsafe path component: {part!r}")
    stamp = pulled_at.astimezone(UTC).strftime(PULLED_AT_FMT)
    return raw_root / source / dataset / key / f"pulled_at={stamp}"


def fresh_pulled_at(raw_root: Path, source: str, dataset: str, key: str) -> datetime:
    """Current UTC time (whole seconds) not yet used by a snapshot of this series.

    Snapshot directories are named to the second; if one exists already, wait for the next
    second rather than altering the recorded pull time.
    """
    while True:
        now = datetime.now(UTC).replace(microsecond=0)
        if not snapshot_dir(raw_root, source, dataset, key, now).exists():
            return now
        time.sleep(1.05 - datetime.now(UTC).microsecond / 1e6)


def write_snapshot(
    df: pd.DataFrame,
    *,
    raw_root: Path,
    source: str,
    dataset: str,
    key: str,
    endpoint: str,
    query: dict[str, Any],
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
    pulled_at: datetime,
    validation: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Snapshot:
    """Write a snapshot and its manifest. Fails if the snapshot already exists."""
    path = snapshot_dir(raw_root, source, dataset, key, pulled_at)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError as e:
        raise SnapshotExistsError(str(path)) from e

    data_file = path / "part-0.parquet"
    df.to_parquet(data_file, index=False)
    sha, dirty = git_info()
    manifest: dict[str, Any] = {
        "source": source,
        "dataset": dataset,
        "key": key,
        "endpoint": endpoint,
        "query": query,
        "requested_start": requested_start.isoformat(),
        "requested_end": requested_end.isoformat(),
        "pulled_at": pulled_at.astimezone(UTC).isoformat(),
        "row_count": len(df),
        "columns": [str(c) for c in df.columns],
        "data_start": _ts_or_none(df, "min"),
        "data_end": _ts_or_none(df, "max"),
        "content_sha256": hashlib.sha256(data_file.read_bytes()).hexdigest(),
        "data_sha256": data_hash(df),
        "git_sha": sha,
        "git_dirty": dirty,
        "library_versions": library_versions(),
        "validation": validation,
        **(extra or {}),
    }
    manifest_file = path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    for f in (data_file, manifest_file):
        os.chmod(f, 0o444)
    return Snapshot(path, manifest)


def _ts_or_none(df: pd.DataFrame, how: str) -> str | None:
    if "timestamp" not in df.columns or df.empty:
        return None
    value = df["timestamp"].min() if how == "min" else df["timestamp"].max()
    return pd.Timestamp(value).isoformat()


def list_snapshots(
    raw_root: Path, source: str, dataset: str, key: str, *, only_valid: bool = True
) -> list[Snapshot]:
    """Snapshots for one series, oldest pull first."""
    base = raw_root / source / dataset / key
    if not base.exists():
        return []
    snaps = []
    for d in sorted(base.glob("pulled_at=*")):
        manifest = json.loads((d / "manifest.json").read_text())
        snap = Snapshot(d, manifest)
        if only_valid and not snap.validation_passed:
            continue
        snaps.append(snap)
    return sorted(snaps, key=lambda s: s.pulled_at)


def load_latest(
    raw_root: Path,
    source: str,
    dataset: str,
    key: str,
    *,
    as_of: datetime | None = None,
    key_cols: tuple[str, ...] = ("timestamp",),
) -> tuple[pd.DataFrame, list[Snapshot]]:
    """Combine valid snapshots: per `key_cols` value, keep the row from the latest pull.

    `as_of` restricts to snapshots pulled at or before that time (vintage reconstruction).
    Returns the combined table and the snapshots used (for lineage logging).
    """
    snaps = list_snapshots(raw_root, source, dataset, key)
    if as_of is not None:
        snaps = [s for s in snaps if s.pulled_at <= as_of]
    if not snaps:
        return pd.DataFrame(), []
    frames = [s.read().assign(_pulled_at=s.pulled_at) for s in snaps]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values([*key_cols, "_pulled_at"]).drop_duplicates(list(key_cols), keep="last")
    return df.drop(columns="_pulled_at").reset_index(drop=True), snaps
