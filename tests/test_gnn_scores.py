"""GNN second opinion: join, bands, coverage, queue column and graph colours."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src import config
from src.gnn_scores import (
    HIGH,
    LOW,
    NOT_SCORED,
    band,
    coverage_problems,
    coverage_report,
    evidence_scores,
    load_gnn_scores,
    second_opinion_text,
    top_percent,
)
from src.ui_data import load_bundle, load_queue
from src.ui_graph import CITATIONS, GNN, blend, build_figure, edge_style, graph_edges


def txn(transaction_id: int, time: str, sender: str, receiver: str) -> dict:
    return {"transaction_id": transaction_id, "hop": 1, "time": time,
            "from_account": sender, "to_account": receiver, "amount_paid": 10.0,
            "payment_currency": "US Dollar", "amount_received": 10.0,
            "receiving_currency": "US Dollar", "payment_format": "ACH"}  # fmt: skip


ALERT = {**txn(1, "2022-09-09 00:03", "A", "B"), "hop": 0}
EVIDENCE = {
    "alert_id": "A0001", "alert_transaction": ALERT,
    "linked_transactions": [txn(2, "2022-09-08 10:00", "B", "C"),   # validation
                            txn(3, "2022-09-05 10:00", "C", "A")],  # training period
    "counts": {"hop1_found": 2, "hop2_found": 0, "included": 2, "truncated": False},
}  # fmt: skip


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    scores = pd.DataFrame({"transaction_id": [1, 2, 50, 51, 52],
                           "score": [0.9, 0.3, 0.1, 0.5, 0.95],
                           "period": ["test", "validation", "test", "test", "test"],
                           "added": [False] * 5})  # fmt: skip
    scores.to_parquet(tmp_path / config.GNN_SCORES_FILE)
    reasons = [{"feature": "sender_sent_7d", "value": 5.0, "shap": 2.0}]
    pd.DataFrame([{"alert_id": "A0001", "rank": 1, "edge_id": 1, "score": 0.99,
                   "payment_format": "ACH", "payment_currency": "US Dollar",
                   "reasons_json": json.dumps(reasons)}]).to_parquet(
        tmp_path / config.ALERTS_FILE)  # fmt: skip
    (tmp_path / config.EVIDENCE_FOLDER).mkdir()
    (tmp_path / config.EVIDENCE_FOLDER / "A0001.json").write_text(json.dumps(EVIDENCE))
    (tmp_path / config.UI_NOTES_FILE).write_text("[]")
    return tmp_path


def test_without_the_scores_file_the_gnn_is_simply_not_available(tmp_path: Path):
    assert load_gnn_scores(tmp_path) is None
    assert second_opinion_text(None, 1) == "GNN second opinion: not available"


def test_bands_use_the_threshold_chosen_on_validation():
    assert band(config.GNN_HIGH_THRESHOLD) == HIGH and band(0.5) == LOW
    assert band(None) == NOT_SCORED


def test_the_header_gives_score_band_and_share_of_test_scored_as_high(folder):
    gnn = load_gnn_scores(folder)
    assert top_percent(gnn, 0.9) == 50.0  # 0.9 and 0.95 of four test scores
    text = second_opinion_text(gnn, 1)
    assert text == "GNN second opinion: 0.90, high (top 50.0% of test)"
    assert top_percent(gnn, 2.0) == 0.0  # above every test score


def test_training_period_rows_get_no_score_and_are_not_a_coverage_problem(folder):
    gnn = load_gnn_scores(folder)
    assert evidence_scores(gnn, EVIDENCE) == {1: 0.9, 2: pytest.approx(0.3)}
    report = coverage_report(gnn, [EVIDENCE])
    assert report["scored"] == 2 and report["training_period_unscored"] == 1
    assert report["problems"] == []


def test_a_scored_period_row_without_a_score_is_reported_as_a_mapping_problem(folder):
    gnn = load_gnn_scores(folder)
    broken = {
        **EVIDENCE,
        "linked_transactions": [txn(99, "2022-09-10 10:00", "B", "C")],
    }
    assert coverage_problems(gnn, broken) == [
        "txn 99 at 2022-09-10 10:00 has no GNN score"]  # fmt: skip


def test_the_queue_shows_the_gnn_score_but_keeps_the_lightgbm_priority(folder):
    row = load_queue(folder, load_gnn_scores(folder)).iloc[0]
    assert row["gnn_score"] == pytest.approx(0.9) and row["priority_rank"] == 1
    assert load_queue(folder).iloc[0]["gnn_score"] is None


def test_the_evidence_table_has_a_gnn_column_blank_for_training_rows(folder):
    bundle = load_bundle(folder, "A0001", load_gnn_scores(folder))
    assert bundle["table"]["gnn_score"].isna().tolist() == [False, False, True]
    assert bundle["second_opinion"].startswith("GNN second opinion: 0.90, high")


def test_gnn_colouring_darkens_with_score_and_thickens_from_the_threshold():
    edges = graph_edges(EVIDENCE, [], {1: 0.9, 2: 0.1})
    high, low, unscored = (edge_style(e, GNN) for e in edges)
    assert high[0] == blend(config.GNN_LOW_COLOUR, config.GNN_HIGH_COLOUR, 0.9)
    assert high[1] > low[1] and unscored == (config.UNSCORED_EDGE_COLOUR,
                                             config.GNN_EDGE_WIDTH, "dash")  # fmt: skip
    assert edge_style(edges[0], CITATIONS)[0] == config.ALERT_EDGE_COLOUR


def test_hover_tells_the_gnn_score_or_why_there_is_none():
    edges = graph_edges(EVIDENCE, [], {1: 0.81})
    assert "GNN 0.81" in edges[0].hover
    assert "not scored (training period)" in edges[2].hover
    assert "GNN" not in graph_edges(EVIDENCE, [])[0].hover


def test_the_gnn_figure_has_a_red_alert_outline_and_a_legend_with_the_threshold():
    figure = build_figure(graph_edges(EVIDENCE, [], {1: 0.9, 2: 0.1}), GNN)
    lines = [t for t in figure.data if t.mode == "lines+markers"]
    assert lines[-2].line.color == config.ALERT_EDGE_COLOUR  # outline under the alert
    bar = [t for t in figure.data if t.mode == "markers" and t.marker.showscale][0]
    assert config.GNN_HIGH_THRESHOLD in bar.marker.colorbar.tickvals
    names = [t.name for t in figure.data if t.showlegend is not False and t.name]
    assert "not scored: GNN training period" in names
    assert not any(t.marker.showscale for t in build_figure(
        graph_edges(EVIDENCE, []), CITATIONS).data if t.mode == "markers")  # fmt: skip
