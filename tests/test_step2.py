"""Tests for step 2: reasons, input, fact check, evaluation (no API calls)."""

from __future__ import annotations

import json

import pandas as pd

from src import config
from src.case_note_input import build_user_message, load_prompt, response_format
from src.evaluate_step2 import evaluate_notes
from src.fact_check import check_note, total_cited
from src.llm_client import cache_key, request_note
from src.reasons import plain_reasons

TRANSACTION = {
    "transaction_id": 1, "from_account": "A", "to_account": "B",
    "amount_paid": 100.0, "amount_received": 100.0, "payment_format": "ACH",
    "payment_currency": "US Dollar",
}  # fmt: skip
OTHER = {**TRANSACTION, "transaction_id": 2, "from_account": "B", "to_account": "C",
         "amount_paid": 50.5, "amount_received": 50.5}  # fmt: skip
EVIDENCE = {
    "alert_transaction": TRANSACTION,
    "linked_transactions": [OTHER],
    "counts": {"hop1_found": 1, "hop2_found": 0},
}
ALERT = {
    "alert_id": "A0001", "score": 0.97, "payment_format": "ACH",
    "payment_currency": "US Dollar",
    "reasons_json": json.dumps([
        {"feature": "payment_format_code", "value": 0.0, "shap": 3.0},
        {"feature": "sender_sent_7d", "value": 5.0, "shap": -1.0},
    ]),
}  # fmt: skip


def good_note(**changes):
    note = {
        "typology": "CYCLE",
        "key_accounts": ["A"],
        "cited_txn_ids": [1, 2],
        "total_amount": 150.5,
        "red_flags": ["A paid 100.00 (txn 1)"],
        "investigator_questions": ["Why did A pay B (txn 1)?"],
        "reason": "ok",
    }
    note.update(changes)
    return note


def test_reasons_use_the_alerts_own_text_not_codes():
    first, second = plain_reasons(ALERT)
    assert "ACH" in first and "raises" in first
    assert "5 payments in the last 7 days" in second and "lowers" in second


def test_user_message_has_no_label_or_pattern_words():
    message = build_user_message(ALERT, EVIDENCE).lower()
    assert "laundering" not in message and "pattern" not in message
    assert '"transaction_id": 2' in message


def test_prompt_file_exists_and_names_every_typology():
    for version in config.PROMPT_VERSIONS:
        prompt = load_prompt(version)
        assert all(name in prompt for name in config.TYPOLOGIES)


def test_schema_requires_every_output_field():
    schema = response_format()["json_schema"]["schema"]
    assert set(schema["required"]) == set(schema["properties"])


def test_fact_check_accepts_a_note_made_only_of_evidence():
    assert check_note(good_note(), EVIDENCE) == []


def test_fact_check_flags_an_unknown_transaction_id():
    assert any(
        "999" in v for v in check_note(good_note(cited_txn_ids=[1, 999]), EVIDENCE)
    )


def test_fact_check_flags_an_unknown_account():
    assert check_note(good_note(key_accounts=["ZZZ"]), EVIDENCE)


def test_total_is_computed_from_cited_transactions():
    by_id = {t["transaction_id"]: t for t in [TRANSACTION, OTHER]}
    assert total_cited(good_note(), by_id) == 150.5


def test_schema_has_no_total_for_the_model_to_get_wrong():
    schema = response_format()["json_schema"]["schema"]
    assert "total_amount" not in schema["properties"]


def test_fact_check_flags_an_amount_in_text_not_in_evidence():
    note = good_note(red_flags=["A paid 777.77 (txn 1)"])
    assert any("777.77" in v for v in check_note(note, EVIDENCE))


def test_cache_key_changes_with_input():
    assert cache_key("a", "v1") != cache_key("b", "v1")
    assert cache_key("a", "v1") != cache_key("a", "v2")


def test_cached_answer_is_returned_without_calling_the_model(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LLM_CACHE_DIR", tmp_path)
    (tmp_path / (cache_key("hello", "v2") + ".json")).write_text(
        json.dumps({"content": "{}"})
    )
    result = request_note(client=None, user_message="hello", version="v2")
    assert result["from_cache"] is True


def test_evaluation_scores_typology_review_flags_and_invented_facts():
    records = [
        {"alert_id": "A1", "note": good_note(), "violations": [], "claims": 4,
         "input_tokens": 1000, "output_tokens": 100, "seconds": 2.0,
         "needs_review": False},
        {"alert_id": "A2", "note": good_note(typology="FAN_IN"),
         "violations": ["account Z not in evidence"], "claims": 4,
         "input_tokens": 1000, "output_tokens": 100, "seconds": 4.0,
         "needs_review": True},
        {"alert_id": "A3", "note": None, "violations": [], "claims": 0,
         "input_tokens": 1000, "output_tokens": 0, "seconds": 1.0,
         "needs_review": True},
    ]  # fmt: skip
    truth = pd.DataFrame({"alert_id": ["A1", "A2", "A3"], "is_laundering": [1, 1, 0],
                          "pattern_type": ["CYCLE", "FAN_OUT", None]})  # fmt: skip
    report = evaluate_notes(records, truth, "v2")
    assert report["json_valid"] == 2
    assert report["notes_needing_review"] == 2
    assert report["typology"]["accuracy"] == 0.5
    assert report["notes_with_invented_facts"] == 1
    assert report["mean_seconds_per_note"] == round(7 / 3, 2)
