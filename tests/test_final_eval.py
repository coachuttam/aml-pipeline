"""Fixed fact check (one-decimal amounts, dates), once-only test guard, baselines."""

from __future__ import annotations

from pathlib import Path

import pytest

from src import config
from src.baselines import typology_baselines
from src.fact_check import check_note
from src.score_student import COMPARISON_FILE, refuse_test_rescore


def payment(transaction_id: int, sender: str, receiver: str, amount: float) -> dict:
    return {"transaction_id": transaction_id, "hop": 1, "time": "2022-09-07 10:00",
            "from_account": sender, "to_account": receiver, "amount_paid": amount,
            "payment_currency": "Yen", "amount_received": amount,
            "receiving_currency": "Yen", "payment_format": "ACH"}  # fmt: skip


EVIDENCE = {
    "alert_transaction": {**payment(1, "S", "R", 1996968.91), "hop": 0},
    "linked_transactions": [payment(2, "R", "X", 13775.6)],
}


def note(text: str) -> dict:
    return {"typology": "unclear", "key_accounts": ["S"], "cited_txn_ids": [1],
            "red_flags": [text], "investigator_questions": [],
            "reason": ""}  # fmt: skip


def test_an_invented_amount_with_one_decimal_is_now_caught():
    violations = check_note(note("S pays 19,969,689.1 Yen (txn 1)."), EVIDENCE)
    assert violations == ["amount 19969689.10 in text not in evidence"]


def test_a_real_amount_with_its_trailing_zero_dropped_passes():
    assert check_note(note("R pays 13,775.6 Yen (txn 2)."), EVIDENCE) == []


def test_a_scaled_number_like_millions_is_not_read_as_an_amount():
    assert check_note(note("About 2.0 million Yen moves (txn 1)."), EVIDENCE) == []


def test_an_amount_at_the_end_of_a_sentence_is_still_checked():
    assert check_note(note("S pays 55.5."), EVIDENCE)


def test_a_date_that_is_not_an_evidence_date_is_flagged():
    assert check_note(note("On 2022-09-07 S pays R (txn 1)."), EVIDENCE) == []
    violations = check_note(note("On 2022-09-12 S pays R (txn 1)."), EVIDENCE)
    assert violations == ["date 2022-09-12 in text not in evidence"]


def test_test_answers_of_an_older_run_are_refused(tmp_path: Path):
    with pytest.raises(SystemExit):
        refuse_test_rescore(tmp_path / "test_raw_Qwen_evidence1.jsonl", tmp_path)


def test_test_is_scored_only_once(tmp_path: Path):
    raw = tmp_path / ("test_raw_Qwen_%s.jsonl" % config.FROZEN_STUDENT_RUN)
    refuse_test_rescore(raw, tmp_path)  # not scored yet: allowed
    (tmp_path / COMPARISON_FILE).write_text("{}")
    with pytest.raises(SystemExit):
        refuse_test_rescore(raw, tmp_path)


def test_baselines_score_the_constant_and_the_rules_on_the_same_alerts():
    evidence = {"A1": {**EVIDENCE, "alert_id": "A1"}}
    truth = [{"alert_id": "A1", "is_laundering": 1, "pattern_type": "GATHER_SCATTER",
              "pattern_visible": True}]  # fmt: skip
    report = typology_baselines(evidence, truth)
    assert report["always_gather_scatter"]["typology_right_visible"] == 1
    assert report["rules"]["visible"] == 1
