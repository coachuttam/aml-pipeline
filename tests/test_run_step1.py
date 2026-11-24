"""End-to-end test of step 1 on the small made-up CSV."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src import config
from src.run_step1 import run


@pytest.fixture
def finished(tiny_csv: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(config, "MAX_TREES", 20)
    monkeypatch.setattr(config, "TOP_ALERTS", 25)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "PATTERNS_FILE", tmp_path / "no-patterns.txt")
    monkeypatch.setattr(config, "PHASE_E_RECORD", tmp_path / "no-record.json")
    output = tmp_path / "artifacts"
    return output, run(output, tiny_csv)


def test_step_one_writes_alerts_evidence_ground_truth_and_a_summary(finished) -> None:
    output, summary = finished
    assert (output / "alerts.parquet").exists()
    assert (output / "ground_truth.parquet").exists()
    assert (output / "step1_summary.json").exists()
    assert summary["alerts"] == 25


def test_there_is_one_evidence_file_per_alert(finished) -> None:
    output, _ = finished
    alerts = pd.read_parquet(output / "alerts.parquet")
    files = sorted(path.stem for path in (output / "evidence").glob("*.json"))
    assert files == sorted(alerts["alert_id"])


def test_alerts_and_evidence_hold_no_label_but_ground_truth_does(finished) -> None:
    output, _ = finished
    alerts = pd.read_parquet(output / "alerts.parquet")
    assert not [c for c in alerts.columns if "laundering" in c]
    for path in (output / "evidence").glob("*.json"):
        assert "laundering" not in path.read_text().lower()
    truth = pd.read_parquet(output / "ground_truth.parquet")
    assert "is_laundering" in truth.columns and len(truth) == len(alerts)
    assert truth["alert_id"].tolist() == alerts["alert_id"].tolist()


def test_a_missing_patterns_file_is_reported_in_the_summary(finished) -> None:
    output, summary = finished
    assert summary["pattern_types_available"] is False
    saved = json.loads((output / "step1_summary.json").read_text())
    assert saved["pattern_types_available"] is False
    assert saved["matches_phase_e_record"]["matches"] is False  # no record in the test


def test_the_laundering_share_matches_the_ground_truth_table(finished) -> None:
    output, summary = finished
    truth = pd.read_parquet(output / "ground_truth.parquet")
    assert summary["laundering_share_of_alerts"] == pytest.approx(
        truth["is_laundering"].mean()
    )
