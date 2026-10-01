from pathlib import Path

import mlflow
import pytest

from pricefc.config import load_config
from pricefc.tracking.mlflow_utils import EXPERIMENTS, base_tags, setup_tracking, start_run


def test_run_logs_required_tags(tmp_path: Path) -> None:
    cfg = load_config(
        Path("configs/base.yaml"),
        {
            "mlflow": {
                "tracking_uri": f"sqlite:///{tmp_path}/m.db",
                "artifact_root": str(tmp_path / "art"),
            }
        },
    )
    setup_tracking(cfg)
    for name in EXPERIMENTS:
        assert mlflow.get_experiment_by_name(name) is not None
    tags = base_tags(cfg, zone="SE3", pipeline_stage="test")
    with start_run("training", tags, cfg) as run:
        pass
    got = mlflow.get_run(run.info.run_id).data.tags
    assert got["zone"] == "SE3" and got["config_hash"] == tags["config_hash"]


def test_missing_tags_rejected() -> None:
    cfg = load_config(Path("configs/base.yaml"))
    with pytest.raises(ValueError, match="missing required"), start_run("training", {}, cfg):
        pass
