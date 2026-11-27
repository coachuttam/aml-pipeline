"""Recall per typology: GNN against LightGBM, on made-up transactions."""

from __future__ import annotations

import pandas as pd

from src import config
from src.typology_recall import (
    laundering_table,
    markdown,
    recall_by_typology,
    type_by_edge,
)


def made_up_edges() -> pd.DataFrame:
    return pd.DataFrame({
        "transaction_id": [1, 2, 3, 4, 5, 6],
        "is_laundering": [1, 1, 1, 1, 1, 0],
        "lightgbm": [0.9, 0.1, 0.5, 0.0, 0.3, 0.99],
        "gnn": [0.1, 0.8, 0.9, 0.0, 0.0, 0.99],
    })  # fmt: skip


def test_type_by_edge_gives_each_pattern_member_its_attempts_type():
    patterns = {("k1",): ("FAN_IN", 0), ("k2",): ("SCATTER_GATHER", 1)}
    assert type_by_edge(patterns, {0: {1, 2}, 1: {3}}) == {
        1: "FAN_IN", 2: "FAN_IN", 3: "SCATTER_GATHER"}  # fmt: skip


def test_laundering_table_drops_normal_rows_and_marks_unlisted_ones():
    table = laundering_table(made_up_edges(), {1: "FAN_IN", 2: "FAN_IN"})
    assert list(table["transaction_id"]) == [1, 2, 3, 4, 5]
    assert list(table["typology"])[2:] == [config.UNLISTED] * 3


def test_recall_counts_each_model_at_its_own_threshold_and_either():
    table = laundering_table(made_up_edges(), {1: "FAN_IN", 2: "FAN_IN", 3: "CYCLE"})
    rows = {r["typology"]: r for r in recall_by_typology(table, 0.221, 0.698)}
    assert (rows["FAN_IN"]["lightgbm_caught"], rows["FAN_IN"]["gnn_caught"]) == (1, 1)
    assert rows["FAN_IN"]["either_caught"] == 2
    assert rows["ALL"]["n"] == 5 and rows["ALL"]["lightgbm_caught"] == 3
    assert rows["ALL"]["gnn_caught"] == 2


def test_focus_typologies_come_first_and_all_comes_last():
    table = laundering_table(made_up_edges(), {1: "CYCLE", 2: "FAN_OUT"})
    names = [r["typology"] for r in recall_by_typology(table, 0.5, 0.5)]
    assert names == ["FAN_OUT", "CYCLE", config.UNLISTED, "ALL"]


def test_markdown_shows_counts_and_percentages():
    table = laundering_table(made_up_edges(), {1: "FAN_IN"})
    text = markdown(recall_by_typology(table, 0.221, 0.698))
    assert "| **FAN_IN** | 1 | 1 (100%) | 0 (0%) | 1 (100%) |" in text
