"""Tests for src/data.py (row order, split, labels kept apart)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data import DaySplit, day_split, load_dataset, load_in_multignn_order
from tests.conftest import raw_line, write_raw


def test_ten_equal_days_split_six_two_two() -> None:
    seconds = np.repeat(np.arange(10) * 86400, 10)
    split = day_split(seconds)
    assert (split.train_end, split.validation_end, split.total) == (60, 80, 100)


def test_the_split_cuts_between_whole_days_never_inside_a_day() -> None:
    counts = [30, 5, 5, 20, 20, 10, 5, 5]
    seconds = np.concatenate([np.full(c, d * 86400 + 5) for d, c in enumerate(counts)])
    split = day_split(seconds)
    edges = np.cumsum([0] + counts)
    assert split.train_end in edges and split.validation_end in edges


def test_the_period_slices_cover_every_row_once() -> None:
    split = DaySplit(6, 8, 10)
    rows = list(range(10))
    assert rows[split.train] + rows[split.validation] + rows[split.test] == rows


def test_rows_come_back_in_time_order_with_ties_in_file_order(tmp_path: Path) -> None:
    path = write_raw(
        tmp_path / "raw.csv",
        [raw_line(30, "a", "b"), raw_line(10, "c", "d"), raw_line(30, "e", "f")],
    )
    frame = load_in_multignn_order(path)
    assert list(frame["from_account"]) == ["c", "a", "e"]
    assert frame["seconds"].tolist() == [0, 1200, 1200]


def test_edge_id_is_the_row_of_the_raw_csv(tmp_path: Path) -> None:
    path = write_raw(
        tmp_path / "raw.csv", [raw_line(30, "a", "b"), raw_line(10, "c", "d")]
    )
    frame = load_in_multignn_order(path)
    assert frame["edge_id"].tolist() == [1, 0]  # sorted by time, ids from the file


def test_the_transactions_have_no_label_column_and_the_labels_are_apart(
    tiny_csv: Path,
) -> None:
    dataset = load_dataset(tiny_csv)
    assert "is_laundering" not in dataset.transactions.columns
    assert len(dataset.truth) == len(dataset.transactions) and dataset.truth.sum() > 0
    assert dataset.split.total == len(dataset.transactions)
