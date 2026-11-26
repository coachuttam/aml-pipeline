"""The student input without model score and push sizes, cue matching, reuse."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src import config
from src.case_note_input import (
    build_blind_user_message,
    build_user_message,
    load_prompt,
    user_message_for,
)
from src.finetune_cues import cue_profile
from src.finetune_dataset import make_record
from src.finetune_reuse import can_reuse, load_old_notes
from src.finetune_select import select_alerts, take_by_weight
from src.score_student import student_record
from src.student_generate import is_valid, needs_retry
from tests.test_finetune import ALERT, EVIDENCE, good_note

FIELDS = list(good_note())


def test_blind_input_has_no_score_and_no_push_sizes():
    message = build_blind_user_message(ALERT, EVIDENCE)
    text = message.lower()
    assert "model_score" not in text and "score" not in text
    assert "raises" not in text and "lowers" not in text
    assert json.loads(message)["reasons"] == [
        "The sender made 5 payments in the last 7 days."
    ]


def test_blind_input_keeps_the_same_transactions_as_the_scored_input():
    scored = json.loads(build_user_message(ALERT, EVIDENCE))
    blind = json.loads(build_blind_user_message(ALERT, EVIDENCE))
    assert blind["transactions"] == scored["transactions"]
    assert blind["alert_transaction_id"] == scored["alert_transaction_id"]


def test_prompt_v4_uses_the_blind_input_and_v3_the_scored_one():
    assert "model_score" in user_message_for("v3", ALERT, EVIDENCE)
    assert "model_score" not in user_message_for("v4", ALERT, EVIDENCE)
    assert "model score" not in load_prompt("v4").lower()
    assert config.STUDENT_PROMPT_VERSION in config.BLIND_PROMPT_VERSIONS


def pool_table() -> pd.DataFrame:
    """Made-up pool: 60 laundering and 60 false alarms, each class 50/50 by feature."""
    rows = []
    for index in range(120):
        feature = "sender_sent_7d" if index % 2 else "payment_format_code"
        laundering = index < 60
        rows.append({
            "alert_id": "P%03d" % index, "is_laundering": int(laundering),
            "pattern_type": "CYCLE" if laundering else None,
            "attempt_id": index // 2 if laundering else None,
            "score": 1.0 - index / 1000, "from_account": "S%d" % index,
            "reasons_json": json.dumps([{"feature": feature, "value": 1, "shap": 1}]),
        })  # fmt: skip
    return pd.DataFrame(rows)


def test_each_class_gets_the_top_reason_mix_of_the_weights(monkeypatch):
    monkeypatch.setattr(config, "FINETUNE_ALERTS", 40)
    monkeypatch.setattr(config, "FINETUNE_ALERTS_PER_PATTERN", 5)
    weights = {"sender_sent_7d": 0.75, "payment_format_code": 0.25}
    chosen = select_alerts(pool_table(), weights)
    for label in (0, 1):
        part = chosen[chosen["is_laundering"] == label]
        features = part["reasons_json"].map(lambda r: json.loads(r)[0]["feature"])
        assert (features == "sender_sent_7d").mean() == pytest.approx(0.75, abs=0.05)


def test_missing_rows_of_a_feature_are_filled_from_the_best_other_rows():
    ordered = pool_table().assign(top_feature="payment_format_code")
    chosen = take_by_weight(ordered, 10, {"sender_sent_7d": 0.8})
    assert len(chosen) == 10


def test_cue_profile_counts_score_bins_top_reasons_and_laundering_share():
    table = pool_table().head(4)
    alerts = table.drop(columns=["is_laundering"]).assign(
        score=[1.0, 1.0, 0.7, 0.2],
        payment_format="ACH",
        payment_currency="US Dollar",
    )
    profile = cue_profile(alerts, table[["alert_id", "is_laundering"]])
    assert profile["alerts"] == 4 and profile["laundering_share"] == 1.0
    assert profile["score_bin_share"]["saturated (1.0)"] == 0.5
    assert profile["top_feature_share"] == {
        "payment_format_code": 0.5,
        "sender_sent_7d": 0.5,
    }


def test_a_note_that_mentions_the_score_is_not_reused():
    assert can_reuse(good_note())
    assert not can_reuse(good_note(reason="The model score of 1.0 is high."))
    assert not can_reuse(
        good_note(red_flags=["The ACH format raises the score (txn 1)"])
    )


def test_old_notes_are_found_by_transaction_id(tmp_path: Path):
    candidates, old = tmp_path / "c", tmp_path / "o"
    candidates.mkdir(), old.mkdir()
    pd.DataFrame({"alert_id": ["A0001"], "edge_id": [77]}).to_parquet(
        candidates / config.GROUND_TRUTH_FILE
    )
    user = json.dumps({"alert_id": "A0001"})
    example = {"messages": [{"content": "s"}, {"content": user},
                            {"content": json.dumps(good_note())}]}  # fmt: skip
    (old / "train_full.jsonl").write_text(json.dumps(example) + "\n")
    assert load_old_notes(old, candidates) == {77: good_note()}


class NoCalls:
    """A client that fails the test if the teacher is called."""

    @property
    def chat(self):
        raise AssertionError("the teacher must not be called for a reused note")


def test_make_record_reuses_an_old_note_without_calling_the_teacher():
    row = {"split": "train", "is_laundering": 1, "pattern_type": "CYCLE", "edge_id": 5,
           "pattern_edge_ids": json.dumps([1, 2, 3])}  # fmt: skip
    record = make_record(NoCalls(), ALERT, EVIDENCE, row, {5: good_note()})
    assert record["reused"] and record["rejected"] == []
    assert record["input_tokens"] == 0 and record["output_tokens"] == 0


def test_a_truncated_or_incomplete_answer_is_invalid_and_retried_once():
    good = {"text": json.dumps(good_note()), "truncated": False}
    assert is_valid(good) and not needs_retry(good)
    cut = {"text": json.dumps(good_note()), "truncated": True}
    assert not is_valid(cut) and needs_retry(cut)
    assert not needs_retry({**cut, "retried": True})
    assert not is_valid({"text": json.dumps({"typology": "CYCLE"}), "truncated": False})
    assert not is_valid({"text": "not json", "truncated": False})
    assert set(json.loads(good["text"])) == set(FIELDS)


def test_a_truncated_student_answer_has_no_note():
    alert = {"alert_id": "A0001", "rank": 1, "score": 1.0}
    raw = {"text": json.dumps(good_note()), "truncated": True,
           "input_tokens": 1, "output_tokens": 1}  # fmt: skip
    assert student_record(alert, EVIDENCE, raw)["note"] is None
