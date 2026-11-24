"""Tests for the UI loaders, the graph edges and the audit log (made-up files)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src import config
from src.audit_log import latest_decision_by_alert, read_decisions, record_decision
from src.ui_data import cited_ids, load_bundle, load_queue
from src.ui_graph import graph_edges

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
    note = {"typology": "FAN_OUT", "cited_txn_ids": [1, 2]}
    record = {
        "alert_id": "A0001",
        "note": note,
        "needs_review": False,
        "violations": [],
    }
    (tmp_path / config.UI_NOTES_FILE).write_text(json.dumps([record]))
    return tmp_path


def test_queue_has_priority_score_top_reason_typology_and_review_flag(folder):
    row = load_queue(folder).iloc[0]
    assert row["priority_rank"] == 1 and row["typology_hint"] == "FAN_OUT"
    assert "7 days" in row["top_reason"] and row["needs_review"] == False  # noqa: E712


def test_a_missing_note_shows_as_needing_review(folder):
    (folder / config.UI_NOTES_FILE).write_text("[]")
    row = load_queue(folder).iloc[0]
    assert row["typology_hint"] == "no note" and row["needs_review"]


def test_bundle_has_reasons_evidence_table_and_cited_ids(folder):
    bundle = load_bundle(folder, "A0001")
    assert len(bundle["table"]) == 2 and bundle["reasons"]
    assert cited_ids(bundle["record"]) == [1, 2]


def test_bundle_and_queue_carry_no_label_column(folder):
    assert not [c for c in load_queue(folder).columns if "launder" in c]
    assert not [
        c for c in load_bundle(folder, "A0001")["table"].columns if "launder" in c
    ]


def test_graph_marks_only_the_alert_transaction(folder):
    edges = graph_edges(EVIDENCE)
    assert [e[3] for e in edges] == [True, False]
    assert edges[1][:2] == ("B", "C")


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
