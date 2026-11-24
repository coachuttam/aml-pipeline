"""Tests for src/ground_truth.py (labels and pattern types, evaluation only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_in_multignn_order
from src.ground_truth import (
    ground_truth_table,
    key_of_row,
    laundering_share,
    read_patterns,
    write_ground_truth,
)
from tests.conftest import raw_line, write_raw

PATTERNS = """INJECTING LI-Small PATTERNS
BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out
2022/09/02 00:10,10,S,10,R,250.50,US Dollar,250.50,US Dollar,Cheque,1
2022/09/02 00:20,10,S,10,T,100.00,US Dollar,100.00,US Dollar,Cheque,1
END LAUNDERING ATTEMPT - FAN-OUT
BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 4 hops
2022/09/03 01:00,10,A,10,B,75.00,US Dollar,75.00,US Dollar,Cheque,1
END LAUNDERING ATTEMPT - CYCLE
"""


def small_table(tmp_path: Path):
    lines = [
        raw_line(24 * 60 + 10, "S", "R", 250.5, 1),
        raw_line(24 * 60 + 11, "P", "Q", 10.0, 0),
        raw_line(2 * 24 * 60 + 60, "A", "B", 75.0, 1),
    ]
    frame = load_in_multignn_order(write_raw(tmp_path / "t.csv", lines))
    truth = frame["is_laundering"].to_numpy()
    return frame.drop(columns=["is_laundering"]), truth


def alerts_for(transactions) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "alert_id": ["A0001", "A0002", "A0003"],
            "edge_id": transactions["edge_id"].to_numpy(),
            "row": np.arange(len(transactions)),
        }
    )


def test_a_missing_patterns_file_gives_no_patterns(tmp_path: Path) -> None:
    assert read_patterns(tmp_path / "LI-Small_Patterns.txt") == {}


def test_each_listed_transaction_gets_its_pattern_type_and_attempt(
    tmp_path: Path,
) -> None:
    path = tmp_path / "patterns.txt"
    path.write_text(PATTERNS)
    patterns = read_patterns(path)
    assert len(patterns) == 3
    assert sorted(set(kind for kind, _ in patterns.values())) == ["CYCLE", "FAN_OUT"]
    assert sorted(attempt for _, attempt in patterns.values()) == [0, 0, 1]


def test_the_table_has_the_label_and_pattern_for_each_alert(tmp_path: Path) -> None:
    path = tmp_path / "patterns.txt"
    path.write_text(PATTERNS)
    transactions, truth = small_table(tmp_path)
    table = ground_truth_table(
        alerts_for(transactions), transactions, truth, read_patterns(path)
    )
    by_id = table.set_index("alert_id")
    assert by_id.loc["A0001", "pattern_type"] == "FAN_OUT"
    assert by_id.loc["A0003", "pattern_type"] == "CYCLE"
    assert pd.isna(by_id.loc["A0002", "pattern_type"])
    assert table["is_laundering"].tolist() == [1, 0, 1]


def test_without_a_patterns_file_the_label_is_kept_and_the_type_is_empty(
    tmp_path: Path,
) -> None:
    transactions, truth = small_table(tmp_path)
    table = ground_truth_table(alerts_for(transactions), transactions, truth, {})
    assert table["is_laundering"].sum() == 2 and table["pattern_type"].isna().all()


def test_the_key_matches_the_pattern_files_format(tmp_path: Path) -> None:
    transactions, _ = small_table(tmp_path)
    key = key_of_row(transactions, 0)
    assert key[0] == "2022/09/02 00:10" and key[5] == 250.5


def test_the_laundering_share_is_the_mean_of_the_labels() -> None:
    table = pd.DataFrame({"is_laundering": [1, 0, 1, 1]})
    assert laundering_share(table) == 0.75


def test_ground_truth_is_written_to_its_own_parquet_file(tmp_path: Path) -> None:
    table = pd.DataFrame({"alert_id": ["A0001"], "is_laundering": [1]})
    path = write_ground_truth(table, tmp_path)
    assert path.name == "ground_truth.parquet" and path.exists()
