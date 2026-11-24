"""Tests for src/reproduce.py."""

from __future__ import annotations

import json
from pathlib import Path

from src.reproduce import compare_with_phase_e


def write_record(path: Path, f1: float, pr_auc: float) -> Path:
    path.write_text(
        json.dumps({"metrics": {"scored": {"minority_f1": f1, "pr_auc": pr_auc}}})
    )
    return path


def test_equal_scores_match(tmp_path: Path) -> None:
    record = write_record(tmp_path / "r.json", 0.2155, 0.118)
    check = compare_with_phase_e({"minority_f1": 0.2155, "pr_auc": 0.118}, record)
    assert check["matches"] is True


def test_a_gap_above_the_tolerance_is_reported_not_hidden(tmp_path: Path) -> None:
    record = write_record(tmp_path / "r.json", 0.2155, 0.118)
    check = compare_with_phase_e({"minority_f1": 0.30, "pr_auc": 0.118}, record)
    assert check["matches"] is False and check["gaps"]["minority_f1"] > 0.08


def test_a_missing_record_does_not_match(tmp_path: Path) -> None:
    check = compare_with_phase_e(
        {"minority_f1": 0.2, "pr_auc": 0.1}, tmp_path / "x.json"
    )
    assert check["matches"] is False and "not found" in check["reason"]
