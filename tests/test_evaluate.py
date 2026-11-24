"""Tests for src/evaluate.py and pattern_members (evaluation only)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import load_in_multignn_order
from src.evaluate import coverage_summary, evidence_ids, mean_coverage, pattern_coverage
from src.ground_truth import pattern_members, read_patterns
from tests.conftest import raw_line, write_raw


def document(alert_edge: int, linked) -> dict:
    return {
        "alert_id": "A0001",
        "alert_transaction": {"transaction_id": alert_edge},
        "linked_transactions": [{"transaction_id": i} for i in linked],
    }


def truth(attempt, laundering: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "alert_id": ["A0001"],
            "is_laundering": [laundering],
            "pattern_type": ["FAN_OUT" if attempt is not None else None],
            "attempt_id": [attempt],
        }
    )


def test_the_alert_transaction_counts_as_found() -> None:
    assert evidence_ids(document(7, [8, 9])) == {7, 8, 9}


def test_coverage_is_the_share_of_the_pattern_found_in_the_evidence() -> None:
    members = {3: {7, 8, 9, 10}}
    table = pattern_coverage([document(7, [8, 99])], truth(3), members)
    assert table["found"].tolist() == [2] and table["coverage"].tolist() == [0.5]
    assert mean_coverage(table) == 0.5


def test_only_laundering_alerts_with_a_known_pattern_are_scored() -> None:
    members = {3: {7, 8}}
    assert (
        len(pattern_coverage([document(7, [])], truth(3, laundering=0), members)) == 0
    )
    assert len(pattern_coverage([document(7, [])], truth(None), members)) == 0
    assert mean_coverage(pattern_coverage([], truth(None), members)) == 0.0


def test_the_summary_counts_full_coverage() -> None:
    members = {3: {7, 8}}
    table = pattern_coverage([document(7, [8])], truth(3), members)
    summary = coverage_summary(table)
    assert summary["alerts_with_full_coverage"] == 1 and summary["mean_coverage"] == 1.0


def test_pattern_members_find_every_listed_transaction_in_the_data(
    tmp_path: Path,
) -> None:
    lines = [
        raw_line(24 * 60 + 10, "S", "R", 250.5, 1),
        raw_line(24 * 60 + 20, "S", "T", 100.0, 1),
        raw_line(24 * 60 + 30, "P", "Q", 10.0, 0),
    ]
    frame = load_in_multignn_order(write_raw(tmp_path / "t.csv", lines))
    path = tmp_path / "patterns.txt"
    path.write_text(
        "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 2-degree\n"
        "2022/09/02 00:10,10,S,10,R,250.50,US Dollar,250.50,US Dollar,Cheque,1\n"
        "2022/09/02 00:20,10,S,10,T,100.00,US Dollar,100.00,US Dollar,Cheque,1\n"
        "END LAUNDERING ATTEMPT - FAN-OUT\n"
    )
    members = pattern_members(
        read_patterns(path), frame.drop(columns=["is_laundering"])
    )
    assert members == {0: {0, 1}}


def test_no_patterns_means_no_members(tmp_path: Path) -> None:
    frame = load_in_multignn_order(
        write_raw(tmp_path / "t.csv", [raw_line(5, "a", "b")])
    )
    assert pattern_members({}, frame) == {}


def test_a_pattern_in_the_data_but_missing_from_the_evidence_scores_zero() -> None:
    table = pattern_coverage([document(7, [])], truth(3), {3: {1, 2, 3}})
    assert table["coverage"].tolist() == [0.0] or table[
        "coverage"
    ].tolist() == pytest.approx([0.0])
