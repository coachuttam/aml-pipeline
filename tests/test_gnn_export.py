"""GNN export: row -> transaction id, every edge scored once, no label written."""

from __future__ import annotations

import numpy as np
import pytest

from src.gnn_export import RECORDED, f1_check, require_all_scored, scores_table


def test_a_formatted_row_maps_to_its_edge_id_the_raw_csv_row():
    edge_ids = np.array([7, 3, 9, 0])  # the formatted file is time-sorted
    table = scores_table({"test": (np.array([2, 0]), np.array([0.9, 0.1]))}, edge_ids)
    assert table["transaction_id"].tolist() == [9, 7]


def test_the_table_has_ids_scores_and_periods_but_no_label():
    periods = {"validation": (np.array([0]), np.array([0.2])),
               "test": (np.array([1]), np.array([0.8]))}  # fmt: skip
    table = scores_table(periods, np.array([5, 6]))
    assert list(table.columns) == ["transaction_id", "score", "period"]
    assert table["period"].tolist() == ["validation", "test"]


def test_an_edge_scored_twice_or_missing_stops_the_export():
    require_all_scored(np.array([3, 1, 2]), np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        require_all_scored(np.array([1, 1, 2]), np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        require_all_scored(np.array([1, 2]), np.array([1, 2, 3]))


def test_recomputed_f1_is_compared_with_the_record_within_a_tolerance():
    close = RECORDED["test_f1"] + 0.005
    assert f1_check({"test_f1": close, "validation_f1": 0.2}) == {
        "test_f1": "matches", "validation_f1": "differs"}  # fmt: skip
