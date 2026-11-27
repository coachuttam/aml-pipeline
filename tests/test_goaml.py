"""goAML-style STR draft: only when escalated, schema-valid, placeholders listed."""

from __future__ import annotations

from pathlib import Path

import pytest

from src import config
from src.audit_log import (
    current_escalation,
    latest_decision_by_alert,
    read_decisions,
    record_decision,
    record_draft,
)
from src.goaml_check import missing_fields, validation_errors
from src.goaml_draft import draft_xml, reason_text


def txn(transaction_id: int, time: str, sender: str, receiver: str) -> dict:
    return {"transaction_id": transaction_id, "hop": 1, "time": time,
            "from_account": sender, "to_account": receiver, "amount_paid": 100.5,
            "payment_currency": "US Dollar", "amount_received": 90.25,
            "receiving_currency": "Euro", "payment_format": "Wire"}  # fmt: skip


ALERT = {"alert_id": "A0007", "rank": 7, "score": 0.98}
EVIDENCE = {
    "alert_id": "A0007",
    "alert_transaction": {**txn(10, "2022-09-09 00:03", "020/80016A780", "011/X"),
                          "hop": 0},
    "linked_transactions": [txn(12, "2022-09-09 05:00", "011/X", "033/Y"),
                            txn(11, "2022-09-09 01:00", "011/X", "044/Z"),
                            txn(13, "2022-09-09 06:00", "033/Y", "055/W")],
}  # fmt: skip
NOTE = {"typology": "FAN_OUT", "key_accounts": ["011/X"],
        "cited_txn_ids": [10, 12, 11, 999], "reason": "011/X fans money out.",
        "red_flags": ["Two payments within hours (txn 11, 12)."],
        "investigator_questions": []}  # fmt: skip
RECORD = {"alert_id": "A0007", "note": NOTE, "needs_review": False, "violations": []}
ESCALATED = {"decision": "escalate", "comment": "Fan-out to new payees",
             "typology_confirmed": ""}  # fmt: skip
WHEN = "2026-11-27T10:00:00"


def draft(escalation: dict = ESCALATED, record: dict = RECORD) -> bytes:
    return draft_xml(ALERT, EVIDENCE, record, escalation, WHEN)


def test_a_draft_is_refused_unless_the_alert_was_escalated():
    with pytest.raises(ValueError):
        draft({**ESCALATED, "decision": "close"})


def test_the_draft_fits_our_schema_and_says_it_is_not_a_certified_fiu_schema():
    xml = draft()
    assert validation_errors(xml) == []
    text = xml.decode()
    assert text.count("not a certified UAE FIU schema") == 2  # comment and element
    assert "<report_code>STR</report_code>" in text


def test_transactions_are_the_alert_then_cited_evidence_rows_by_time():
    text = draft().decode()
    numbers = [line.strip()[19:-20] for line in text.splitlines()
               if "<transactionnumber>" in line]  # fmt: skip
    assert numbers == ["10", "11", "12"]  # 13 not cited, 999 not in the evidence
    assert "<foreign_amount>100.50</foreign_amount>" in text
    assert "<foreign_currency_code>EUR</foreign_currency_code>" in text
    assert "<date_transaction>2022-09-09T00:03:00</date_transaction>" in text


def test_the_typology_goes_in_only_when_the_analyst_confirmed_it():
    assert "Typology FAN_OUT" not in draft().decode()
    confirmed = draft({**ESCALATED, "typology_confirmed": "FAN_OUT"}).decode()
    assert "<indicator>Typology FAN_OUT (analyst confirmed" in confirmed


def test_unknown_fields_are_placeholders_and_listed_once_per_kind():
    gaps = {g["field"]: g for g in missing_fields(draft())}
    assert gaps["SWIFT/BIC"]["count"] == 7  # 3 transactions x 2 sides + 1 party
    assert "reporting entity id (goAML registration)" in gaps
    assert gaps["amount in AED"]["where"] == ["transaction/amount_local"]
    assert "transaction/t_to/to_account/swift" in gaps["SWIFT/BIC"]["where"]


def test_known_values_come_from_the_evidence_and_the_note():
    text = draft().decode()
    assert "<institution_code>020</institution_code>" in text
    assert "<account>80016A780</account>" in text
    assert "<entity_reference>A0007</entity_reference>" in text
    assert text.count("<report_party>") == 1  # the note's one key account


def test_the_reason_has_the_note_the_red_flags_and_the_analyst_comment():
    text = reason_text(ALERT, NOTE, ESCALATED)
    assert "011/X fans money out." in text and "Red flag: Two payments" in text
    assert text.endswith("Analyst comment: Fan-out to new payees")


def test_a_long_reason_is_cut_at_the_limit(monkeypatch):
    monkeypatch.setattr(config, "GOAML_REASON_MAX_CHARS", 50)
    text = reason_text(ALERT, NOTE, ESCALATED)
    assert len(text) == 50 and text.endswith("…")


def test_without_a_note_the_draft_still_fits_and_names_the_alert_sender():
    xml = draft(record={"alert_id": "A0007", "note": None})
    assert validation_errors(xml) == []
    assert xml.decode().count("<transactionnumber>") == 1


def test_the_schema_rejects_a_draft_without_its_reason_or_with_another_report_code():
    text = draft().decode()
    start, end = text.index("<reason>"), text.index("</reason>") + len("</reason>")
    assert validation_errors((text[:start] + text[end:]).encode())
    wrong = text.replace(
        "<report_code>STR</report_code>", "<report_code>CTR</report_code>"
    )
    assert validation_errors(wrong.encode())


def test_only_the_latest_escalation_allows_a_draft_and_drafts_are_logged(
    tmp_path: Path,
):
    log = tmp_path / "audit.jsonl"
    record_decision("A1", "escalate", "why", "v", log, typology_confirmed="CYCLE")
    assert current_escalation("A1", log)["typology_confirmed"] == "CYCLE"
    record_draft("A1", 62, "CYCLE", log)
    assert latest_decision_by_alert(log) == {"A1": "escalate"}  # draft not a decision
    assert read_decisions(log)[-1]["event"] == "goaml_draft"
    record_decision("A1", "close", "false alarm after all", "v", log)
    assert current_escalation("A1", log) is None


def test_only_a_named_typology_is_offered_for_confirmation():
    from src.ui_goaml import suggested_typology

    assert suggested_typology({"record": RECORD}) == "FAN_OUT"
    unclear = {**RECORD, "note": {**NOTE, "typology": config.UNCLEAR}}
    assert suggested_typology({"record": unclear}) is None
    assert suggested_typology({"record": None}) is None
