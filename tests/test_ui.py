"""Tests for the UI loaders, the graph edges and the audit log (made-up files)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src import config
from src.audit_log import latest_decision_by_alert, read_decisions, record_decision
from src.reasons import display_reasons
from src.ui_data import cited_ids, load_bundle, load_queue, reason_line
from src.ui_graph import build_figure, graph_edges

TXN = {
    "transaction_id": 1, "hop": 0, "time": "2022-09-09 00:03", "from_account": "A",
    "to_account": "B", "amount_paid": 10.0, "payment_currency": "US Dollar",
    "amount_received": 10.0, "receiving_currency": "US Dollar", "payment_format": "ACH",
}  # fmt: skip
LINKED = {**TXN, "transaction_id": 2, "hop": 1, "from_account": "B", "to_account": "C"}
EVIDENCE = {
    "alert_id": "A0001", "alert_transaction": TXN, "linked_transactions": [LINKED],
    "counts": {"hop1_found": 1, "hop2_found": 0, "included": 1, "truncated": False},
}  # fmt: skip


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    reasons = [{"feature": "sender_sent_7d", "value": 5.0, "shap": 2.0}]
    alerts = pd.DataFrame([{
        "alert_id": "A0001", "rank": 1, "score": 0.9, "payment_format": "ACH",
        "payment_currency": "US Dollar", "reasons_json": json.dumps(reasons),
    }])  # fmt: skip
    alerts.to_parquet(tmp_path / config.ALERTS_FILE)
    (tmp_path / config.EVIDENCE_FOLDER).mkdir()
    (tmp_path / config.EVIDENCE_FOLDER / "A0001.json").write_text(json.dumps(EVIDENCE))
    note = {
        "typology": "FAN_OUT",
        "cited_txn_ids": [1, 2],
        "red_flags": [],
        "investigator_questions": [],
    }
    record = {
        "alert_id": "A0001",
        "note": note,
        "needs_review": False,
        "violations": [],
    }
    (tmp_path / config.UI_NOTES_FILE).write_text(json.dumps([record]))
    (tmp_path / config.SUMMARY_FILE).write_text(
        json.dumps({"test_transactions": 10000})
    )
    return tmp_path


def test_queue_has_priority_score_top_reason_typology_and_review_flag(folder):
    row = load_queue(folder).iloc[0]
    assert row["priority_rank"] == 1 and row["suggested_typology"] == "FAN_OUT"
    assert "7 days" in row["top_reason"] and "strong" in row["top_reason"]
    assert row["review"] == "✓ clean" and row["score_rank"] == "top 0.0100%"


def test_a_missing_note_shows_as_needing_review(folder):
    (folder / config.UI_NOTES_FILE).write_text("[]")
    row = load_queue(folder).iloc[0]
    assert row["suggested_typology"] == "no note" and row["review"].startswith("⚠")


def test_bundle_has_reasons_evidence_table_and_cited_ids(folder):
    bundle = load_bundle(folder, "A0001")
    assert len(bundle["table"]) == 2 and bundle["reasons"]
    assert cited_ids(bundle["record"]) == [1, 2]


def test_bundle_and_queue_carry_no_label_column(folder):
    assert not [c for c in load_queue(folder).columns if "launder" in c]
    assert not [
        c for c in load_bundle(folder, "A0001")["table"].columns if "launder" in c
    ]


def test_graph_marks_the_alert_and_the_cited_transactions():
    edges = graph_edges(EVIDENCE, cited=[2])
    assert [e.is_alert for e in edges] == [True, False]
    assert [e.is_cited for e in edges] == [False, True]
    assert (edges[1].sender, edges[1].receiver) == ("B", "C")


def test_hover_shows_amount_currency_and_time():
    text = graph_edges(EVIDENCE, [])[0].hover
    assert "10.00 USD" in text and "2022-09-09 00:03" in text


def test_interactive_figure_draws_the_alert_edge_red_and_labels_accounts():
    figure = build_figure(graph_edges(EVIDENCE, [2]))
    colours = [t.line.color for t in figure.data if t.mode == "lines+markers"]
    assert config.ALERT_EDGE_COLOUR in colours and config.CITED_EDGE_COLOUR in colours
    assert set(figure.data[-1].text) == {"A", "B", "C"}


def test_reasons_get_a_strength_label_and_a_share_of_the_top_five():
    reasons = [
        {"feature": "sender_sent_7d", "value": 5.0, "shap": 6.0},
        {"feature": "sender_sent_1h", "value": 1.0, "shap": 3.0},
        {"feature": "sender_sent_24h", "value": 2.0, "shap": -1.0},
    ]
    alert = {"reasons_json": json.dumps(reasons), "payment_format": "ACH",
             "payment_currency": "US Dollar"}  # fmt: skip
    shown = display_reasons(alert)
    assert [r["strength"] for r in shown] == ["strong", "strong", "weak"]
    assert [r["share_percent"] for r in shown] == [60, 30, 10]
    assert shown[2]["direction"] == "lowers"


def test_the_score_phrase_is_gone_from_displayed_reasons(folder):
    assert "raises the score by" not in load_bundle(folder, "A0001")["reasons"][0]


def test_decisions_are_appended_in_order_with_note_version(tmp_path):
    log = tmp_path / "audit.jsonl"
    record_decision("A0001", "close", " false alarm ", "v3", log)
    record_decision("A0001", "escalate", "second look", "v3", log)
    events = read_decisions(log)
    assert [e["decision"] for e in events] == ["close", "escalate"]
    assert events[0]["comment"] == "false alarm" and events[0]["note_version"] == "v3"
    assert latest_decision_by_alert(log) == {"A0001": "escalate"}


def test_a_decision_that_is_not_escalate_or_close_is_refused(tmp_path):
    with pytest.raises(ValueError):
        record_decision("A0001", "maybe", "", "v3", tmp_path / "a.jsonl")


def test_reading_a_missing_log_gives_an_empty_list(tmp_path):
    assert read_decisions(tmp_path / "none.jsonl") == []


def test_ids_written_in_text_are_highlighted_too():
    note = {"cited_txn_ids": [1], "red_flags": ["A paid B (txn 2, 3)"],
            "investigator_questions": ["Why (txn 9)?"]}  # fmt: skip
    assert cited_ids({"note": note}) == [1, 2, 3, 9]


def test_a_rising_reason_has_no_score_phrase_and_a_lowering_one_says_so():
    rising = {
        "text": "T",
        "strength": "strong",
        "share_percent": 50,
        "direction": "raises",
    }
    assert reason_line(rising) == "T — strong, 50% of the top-5 push"
    assert reason_line({**rising, "direction": "lowers"}).endswith("(lowers the score)")


def test_graph_axes_have_padding_beyond_the_account_positions():
    figure = build_figure(graph_edges(EVIDENCE, []))
    xs = [x for t in figure.data[-1:] for x in t.x]
    low, high = figure.layout.xaxis.range
    assert low < min(xs) and high > max(xs)
