"""MLflow helpers enforcing the project's run conventions (spec section 10)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
import yaml

from pricefc.config import BaseConfig, config_hash
from pricefc.lineage import file_hash, git_info

EXPERIMENTS = ("ingest", "datasets", "training", "backtest", "forecast-live")
REQUIRED_TAGS = (
    "zone",
    "resolution",
    "weather_source",
    "dataset_version",
    "git_sha",
    "git_dirty",
    "model_family",
    "compute_env",
    "config_hash",
    "pipeline_stage",
)


def base_tags(
    config: BaseConfig,
    *,
    zone: str,
    pipeline_stage: str,
    model_family: str = "none",
    weather_source: str = "none",
    dataset_version: str = "none",
) -> dict[str, str]:
    sha, dirty = git_info()
    return {
        "zone": zone,
        "resolution": config.resolution,
        "weather_source": weather_source,
        "dataset_version": dataset_version,
        "git_sha": sha,
        "git_dirty": str(dirty).lower(),
        "model_family": model_family,
        "compute_env": config.compute_env,
        "config_hash": config_hash(config),
        "pipeline_stage": pipeline_stage,
        "uv_lock_hash": file_hash(Path("uv.lock")),
    }


def setup_tracking(config: BaseConfig, tracking_uri: str | None = None) -> None:
    mlflow.set_tracking_uri(tracking_uri or config.mlflow.tracking_uri)
    for name in EXPERIMENTS:
        if mlflow.get_experiment_by_name(name) is None:
            mlflow.create_experiment(name, artifact_location=_artifact_location(config, name))


def _artifact_location(config: BaseConfig, experiment: str) -> str | None:
    root = config.mlflow.artifact_root
    if "://" in root:
        return f"{root.rstrip('/')}/{experiment}"
    return Path(root, experiment).resolve().as_uri()


@contextmanager
def start_run(
    experiment: str, tags: dict[str, str], config: BaseConfig, run_name: str | None = None
) -> Iterator[Any]:
    """Start a run with all required tags and the resolved config logged."""
    missing = [t for t in REQUIRED_TAGS if t not in tags]
    if missing:
        raise ValueError(f"missing required MLflow tags: {missing}")
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        mlflow.log_text(yaml.safe_dump(config.model_dump(mode="json")), "config/resolved.yaml")
        yield run
