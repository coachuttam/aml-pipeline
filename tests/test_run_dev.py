"""The dev set comes from the validation period and holds no label."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config
from src.data import load_dataset
from src.run_dev import build_dev_set


def test_dev_alerts_come_from_the_validation_period_and_hold_no_label(
    tiny_csv: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(config, "MAX_TREES", 20)
    monkeypatch.setattr(config, "DEV_ALERTS", 8)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "PATTERNS_FILE", tmp_path / "no-patterns.txt")
    folder = build_dev_set(tmp_path / "artifacts", tiny_csv)
    alerts = pd.read_parquet(folder / "alerts.parquet")
    validation = load_dataset(tiny_csv).split.validation
    assert len(alerts) == 8
    assert alerts["row"].between(validation.start, validation.stop - 1).all()
    assert not [c for c in alerts.columns if "laundering" in c]
    assert len(list((folder / "evidence").glob("*.json"))) == 8
