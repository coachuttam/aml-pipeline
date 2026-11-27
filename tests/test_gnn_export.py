"""GNN export: every period edge scored once, ids mapped, no label written."""

from __future__ import annotations

import numpy as np
import pytest

from src.gnn_export import (
    RECORDED,
    coverage_report,
    f1_check,
    missing_seed_ids,
    require_all_scored,
    scores_table,
    to_batch_nodes,
)


def test_a_formatted_row_maps_to_its_edge_id_the_raw_csv_row():
    edge_ids = np.array([7, 3, 9, 0])  # the formatted file is time-sorted
    periods = {"test": (np.array([2, 0]), np.array([0.9, 0.1]), np.array([0, 1]))}
    table = scores_table(periods, edge_ids)
    assert table["transaction_id"].tolist() == [9, 7]
    assert table["added"].tolist() == [False, True]


def test_the_table_has_ids_scores_periods_and_added_flags_but_no_label():
    periods = {"validation": (np.array([0]), np.array([0.2]), np.array([False])),
               "test": (np.array([1]), np.array([0.8]), np.array([False]))}  # fmt: skip
    table = scores_table(periods, np.array([5, 6]))
    assert list(table.columns) == ["transaction_id", "score", "period", "added"]


def test_the_coverage_report_counts_missing_extra_and_duplicate_edges():
    report = coverage_report(np.array([1, 2, 2, 9]), np.array([1, 2, 3, 4]))
    assert (report["expected"], report["scored"]) == (4, 4)
    assert report["missing"] == 2 and report["missing_examples"] == [3, 4]
    assert report["extra_examples"] == [9] and report["duplicates_examples"] == [2]


def test_the_coverage_report_shows_at_most_five_examples():
    report = coverage_report(np.array([0]), np.arange(20))
    assert report["missing"] == 19 and len(report["missing_examples"]) == 5


def test_an_edge_scored_twice_or_missing_stops_the_export_with_counts():
    assert require_all_scored(np.array([3, 1, 2]), np.array([1, 2, 3]))["missing"] == 0
    with pytest.raises(ValueError, match="'duplicates': 1"):
        require_all_scored(np.array([1, 1, 2, 3]), np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="'missing': 1"):
        require_all_scored(np.array([1, 2]), np.array([1, 2, 3]))


def test_seed_edges_the_sampler_left_out_are_found_so_they_can_be_added_back():
    seeds = np.array([10.0, 11.0, 12.0])  # Multi-GNN keeps ids as float edge features
    sampled = np.array([3.0, 10.0, 12.0, 40.0])  # the batch: seeds and neighbours
    missing = missing_seed_ids(seeds, sampled)
    assert missing.tolist() == [11] and missing.dtype == np.int64


def test_an_added_edge_is_rewired_to_the_batch_positions_of_its_two_accounts():
    batch_nodes = np.array([500, 7, 42])  # global account ids, in batch order
    pairs = np.array([[42, 7], [500, 42]])  # two edges: 42->500 and 7->42
    assert to_batch_nodes(pairs, batch_nodes).tolist() == [[2, 1], [0, 2]]


def test_every_period_edge_is_covered_once_sampled_and_added_edges_together():
    expected = np.arange(100, 110)
    sampled = np.array([100, 101, 103, 104, 106, 107, 109])
    added = missing_seed_ids(expected.astype(float), sampled.astype(float))
    assert (
        require_all_scored(np.concatenate([sampled, added]), expected)["scored"] == 10
    )


def test_recomputed_f1_is_compared_with_the_record_within_a_tolerance():
    close = RECORDED["test_f1"] + 0.005
    assert f1_check({"test_f1": close, "validation_f1": 0.2}) == {
        "test_f1": "matches", "validation_f1": "differs"}  # fmt: skip
