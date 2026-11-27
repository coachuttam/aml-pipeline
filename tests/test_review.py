"""Student review (confusion table, invented facts) and evidence v2 ("capped")."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config
from src.evidence import EvidenceIndex, build_evidence
from src.evidence_v2 import coverage_summary, hop_mix, visible_flags
from src.student_review import (
    FALSE_ALARM,
    INVALID,
    THIN,
    classify_violation,
    confusion_table,
    table_markdown,
    unchecked_amounts,
)
from tests.conftest import raw_line
from tests.test_evidence import DAY, graph_dataset, row_of


def truth(alert_id: str, laundering: int, pattern: str, visible: bool) -> dict:
    return {"alert_id": alert_id, "is_laundering": laundering,
            "pattern_type": pattern, "pattern_visible": visible}  # fmt: skip


def payment(transaction_id: int, amount: float, hop: int = 1) -> dict:
    return {"transaction_id": transaction_id, "hop": hop,
            "from_account": "S", "to_account": "R",
            "amount_paid": amount, "amount_received": amount}  # fmt: skip


def evidence_of(*linked: dict) -> dict:
    return {"alert_id": "A1", "alert_transaction": payment(100, 5.0, hop=0),
            "linked_transactions": list(linked)}  # fmt: skip


def test_each_alert_lands_in_its_true_row_and_predicted_column():
    rows = [truth("A1", 1, "CYCLE", True), truth("A2", 1, "CYCLE", False),
            truth("A3", 0, None, False), truth("A4", 1, "FAN_IN", True)]  # fmt: skip
    table = confusion_table({"A1": "RANDOM", "A2": "unclear", "A3": "unclear"}, rows)
    assert table["CYCLE"] == {"RANDOM": 1}
    assert table[THIN] == {"unclear": 1} and table[FALSE_ALARM] == {"unclear": 1}
    assert table["FAN_IN"] == {INVALID: 1}


def test_the_markdown_table_has_a_total_per_true_row():
    table = {"CYCLE": {"RANDOM": 2, "CYCLE": 1}}
    assert "| CYCLE | 1 | 2 | 3 |" in table_markdown(table)


def test_an_id_one_past_a_run_of_evidence_ids_is_called_extrapolated():
    evidence = evidence_of(payment(201, 1.0), payment(202, 1.0))
    result = classify_violation("txn id 203 in text not in evidence", evidence)
    assert result["kind"] == "transaction id (in text)"
    assert "extrapolated" in result["cause"]


def test_an_evidence_amount_with_its_decimal_point_moved_is_recognised():
    evidence = evidence_of(payment(201, 295474.62))
    result = classify_violation("amount 29547.46 in text not in evidence", evidence)
    assert result["kind"] == "amount" and "decimal point moved" in result["cause"]


def test_a_one_decimal_amount_is_flagged_only_when_not_in_the_evidence():
    note = {"red_flags": ["sends 19,969,689.1 and 13,775.6 Yen"],
            "investigator_questions": [], "reason": ""}  # fmt: skip
    evidence = evidence_of(payment(201, 13775.6), payment(202, 1996968.91))
    assert unchecked_amounts(note, evidence) == ["19,969,689.1"]


def test_capped_evidence_keeps_hop_two_even_when_hop_one_could_fill_everything(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(config, "EVIDENCE_V2_HOP1_EDGES", 2)
    monkeypatch.setattr(config, "EVIDENCE_V2_HOP2_EDGES", 1)
    monkeypatch.setattr(config, "EVIDENCE_MAX_EDGES", 3)
    lines = [raw_line(10 * DAY, "S", "R")]
    lines += [raw_line(10 * DAY + i, "S", "X%d" % i) for i in range(1, 5)]  # hop 1
    lines += [raw_line(10 * DAY + 9, "X1", "Z"), raw_line(10 * DAY + 10, "X2", "Z")]
    dataset = graph_dataset(tmp_path, lines)
    index = EvidenceIndex(dataset.transactions)
    capped = build_evidence(index, row_of(dataset, 0), "A1", "capped")
    nearest = build_evidence(index, row_of(dataset, 0), "A1", "nearest")
    assert [t["hop"] for t in capped["linked_transactions"]] == [1, 1, 2]
    assert [t["hop"] for t in nearest["linked_transactions"]] == [1, 1, 1]
    assert capped["counts"]["truncated"]


def test_hop_mix_counts_rows_per_hop_and_alerts_without_hop_two():
    documents = [evidence_of(payment(1, 1.0), payment(2, 1.0, hop=2)),
                 evidence_of(payment(3, 1.0), payment(4, 1.0))]  # fmt: skip
    mix = hop_mix(documents)
    assert mix["hop_share"] == {"1": 0.75, "2": 0.25}
    assert mix["alerts_without_hop2"] == 1 and mix["mean_linked"] == 2.0


def test_a_pattern_is_visible_from_three_of_its_transactions_in_the_evidence():
    documents = [evidence_of(payment(1, 1.0), payment(2, 1.0), payment(3, 1.0))]
    table = pd.DataFrame([{"alert_id": "A1", "edge_id": 100, "is_laundering": 1,
                           "pattern_type": "CYCLE", "attempt_id": 7}])  # fmt: skip
    members = {7: {1, 2, 3, 4, 5, 6}}
    summary = coverage_summary(documents, table, members)
    assert summary["visible"] == 1 and summary["mean_coverage"] == 0.5
    assert visible_flags(documents, table, members)[0]["pattern_visible"]
