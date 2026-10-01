from pathlib import Path

import pytest
from pydantic import ValidationError

from pricefc.config import config_hash, load_config

BASE = Path("configs/base.yaml")


def test_base_config_loads() -> None:
    cfg = load_config(BASE)
    assert cfg.zones
    assert 0.5 in cfg.quantiles


def test_config_hash_stable_and_sensitive() -> None:
    a = load_config(BASE)
    assert config_hash(a) == config_hash(load_config(BASE))
    b = load_config(BASE, {"seed": a.seed + 1})
    assert config_hash(a) != config_hash(b)


@pytest.mark.parametrize(
    "override",
    [
        {"zones": ["SE5"]},
        {"zones": ["SE3", "SE3"]},
        {"quantiles": [0.1, 0.9]},
        {"quantiles": [0.9, 0.5, 0.1]},
        {"resolution": "daily"},
        {"unknown_key": 1},
    ],
)
def test_invalid_config_rejected(override: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        load_config(BASE, override)
