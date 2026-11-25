"""The V2 evaluation scheme on made-up notes."""

from __future__ import annotations

import pandas as pd

from src.eval_v2 import score_notes
from src.teacher import is_pattern_visible


def record(alert_id: str, typology, violations=0, claims=2) -> dict:
    note = None if typology is None else {"typology": typology}
    return {"alert_id": alert_id, "note": note, "claims": claims,
            "violations": ["x"] * violations}  # fmt: skip


TRUTH = pd.DataFrame(
    {
        "alert_id": ["a", "b", "c", "d", "e"],
        "is_laundering": [1, 1, 1, 0, 0],
        "pattern_type": ["CYCLE", "CYCLE", "STACK", None, None],
        "pattern_visible": [True, True, False, False, False],
    }
)
RECORDS = [
    record("a", "CYCLE"), record("b", "unclear", violations=1),
    record("c", "FAN_IN"), record("d", "unclear"), record("e", None),
]  # fmt: skip


def test_typology_accuracy_counts_only_visible_pattern_laundering():
    report = score_notes(RECORDS, TRUTH)
    assert report["typology_accuracy_visible"] == {"count": 1, "of": 2, "rate": 0.5}


def test_unclear_rates_are_split_by_false_alarm_and_visible_laundering():
    report = score_notes(RECORDS, TRUTH)
    assert report["unclear_rate_false_alarms"]["rate"] == 1.0  # only d is valid
    assert report["unclear_rate_visible_laundering"]["count"] == 1


def test_invented_fact_rate_and_json_validity_use_all_notes():
    report = score_notes(RECORDS, TRUTH)
    assert report["invented_facts_per_claim"] == {"count": 1, "of": 8, "rate": 0.125}
    assert report["json_validity"] == {"count": 4, "of": 5, "rate": 0.8}


def test_a_pattern_is_visible_when_three_of_its_transactions_are_in_the_evidence():
    shown = {"is_laundering": True, "pattern_transaction_ids_in_evidence": [1, 2, 3]}
    assert is_pattern_visible(shown)
    assert not is_pattern_visible({**shown, "pattern_transaction_ids_in_evidence": [1]})
    assert not is_pattern_visible({"is_laundering": False})
