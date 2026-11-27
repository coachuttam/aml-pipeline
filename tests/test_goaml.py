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
from src.goaml_draft import draft_xml, party_accounts, reason_text


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
    assert "amount in AED" not in gaps  # every row pays in USD: AED at the peg
    assert gaps["rate to AED on the day"]["count"] == 3  # the received EUR side
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


def test_usd_rows_get_their_aed_amount_and_rate_at_the_peg_others_stay_missing():
    euro_only = {
        **txn(20, "2022-09-09 02:00", "011/X", "066/V"),
        "payment_currency": "Euro",
    }  # paid and received in euros
    evidence = {**EVIDENCE, "linked_transactions": [euro_only]}
    note = {**NOTE, "cited_txn_ids": [10, 20]}
    text = draft_xml(ALERT, evidence, {**RECORD, "note": note}, ESCALATED, WHEN)
    blocks = text.decode().split("<transaction>")[1:]
    assert "<amount_local>369.09</amount_local>" in blocks[0]  # 100.50 x 3.6725
    assert "AED peg, 3.6725 AED per USD" in blocks[0]
    assert "<foreign_exchange_rate>3.6725</foreign_exchange_rate>" in blocks[0]
    assert "<amount_local>MISSING: amount in AED</amount_local>" in blocks[1]
    assert "peg" not in blocks[1]
    assert validation_errors(text) == []


def parties(typology: str) -> list:
    return [a for a, _, _ in party_accounts(NOTE, EVIDENCE, typology)]


def test_a_confirmed_fan_in_adds_the_cited_senders_as_report_parties():
    assert parties("FAN_IN") == ["011/X", "020/80016A780"]  # 10 from 020; 11, 12 from X


def test_a_confirmed_fan_out_adds_the_cited_receivers_as_report_parties():
    assert parties("FAN_OUT") == ["011/X", "044/Z", "033/Y"]  # by time: 11 then 12


def test_other_or_unconfirmed_typologies_keep_the_key_accounts_only():
    assert parties("CYCLE") == ["011/X"] and parties("") == ["011/X"]
    text = draft({**ESCALATED, "typology_confirmed": "FAN_OUT"}).decode()
    assert text.count("<report_party>") == 3
    assert "Receiver of cited transaction 11 (FAN_OUT, analyst confirmed)" in text


def test_the_saved_state_shows_whether_the_typology_was_confirmed():
    from src.ui_goaml import saved_state_text

    event = {"decision": "escalate", "time": "2026-11-27T10:00:00",
             "typology_confirmed": "FAN_IN"}  # fmt: skip
    assert saved_state_text(event) == (
        "Saved decision: escalate at 2026-11-27 10:00:00 · typology confirmed: FAN_IN"
    )
    assert saved_state_text({**event, "typology_confirmed": ""}).endswith(
        "typology not confirmed")  # fmt: skip


def test_clearing_the_demo_log_archives_it_and_leaves_an_empty_log(tmp_path: Path):
    from src.audit_log import clear_log

    log = tmp_path / "audit_log.jsonl"
    assert clear_log(log) is None  # nothing to clear
    record_decision("A1", "close", "demo", "v", log)
    archived = clear_log(log)
    assert not log.exists() and read_decisions(log) == []
    assert archived.parent.name == config.AUDIT_ARCHIVE_FOLDER
    assert read_decisions(archived)[0]["alert_id"] == "A1"
