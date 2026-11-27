"""goAML-style STR draft for an escalated alert (UAE first). Never filed automatically.

The draft has goAML's main parts: the reporting entity, report code STR, the reason
(from the case note and the analyst's comment), one `transaction` per cited
evidence row (the alert transaction always first), the note's key accounts as
`report_parties`, and `report_indicators`. The typology goes in only if the analyst
confirmed it when escalating.

What the pipeline cannot know (KYC, names, addresses, institution ids, FIU codes,
amounts in AED) is written as "MISSING: <what>", and `missing_fields` lists it for
the UI. Every value that is filled comes from the alert's evidence file, the note or
the analyst. The schema is our own (docs/goaml_style_draft.xsd): a goAML-style draft,
not a certified UAE FIU schema.

Example:
    xml = draft_xml(alert, evidence, record, escalation, "2026-11-27T10:00:00")
    validation_errors(xml)   # [] when it fits the schema
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

from src import config
from src.case_note_input import evidence_transactions
from src.totals import currency_code

MISSING = config.GOAML_MISSING_PREFIX


def missing(what: str) -> str:
    return MISSING + what


def add(parent: ET.Element, tag: str, text: str) -> ET.Element:
    child = ET.SubElement(parent, tag)
    child.text = text
    return child


def split_account(account: str) -> List[str]:
    """ "020/80016A780" -> ["020", "80016A780"] (IBM's bank id, account number)."""
    bank, _, number = account.partition("/")
    return [bank, number] if number else [missing("institution code"), account]


def add_account(parent: ET.Element, tag: str, account: str, currency: str) -> None:
    """An account block: what we know (bank id, number, currency), the rest missing."""
    bank, number = split_account(account)
    block = ET.SubElement(parent, tag)
    add(block, "institution_code", bank)
    add(block, "institution_name", missing("institution name (KYC)"))
    add(block, "swift", missing("SWIFT/BIC"))
    add(block, "account", number)
    add(block, "currency_code", currency_code(currency))
    add(block, "signatory", missing("account holder name (KYC)"))


def add_side(parent: ET.Element, side: str, item: Dict) -> None:
    """t_from or t_to: funds code, original currency and amount, account, country."""
    paid = side == "from"
    block = ET.SubElement(parent, "t_" + side)
    add(block, side + "_funds_code", missing("funds type code (FIU list)"))
    money = ET.SubElement(block, side + "_foreign_currency")
    currency = item["payment_currency" if paid else "receiving_currency"]
    add(money, "foreign_currency_code", currency_code(currency))
    add(
        money,
        "foreign_amount",
        "%.2f" % item["amount_paid" if paid else "amount_received"],
    )
    add(money, "foreign_exchange_rate", missing("rate to AED on the day"))
    add_account(block, side + "_account", item[side + "_account"], currency)
    add(block, side + "_country", missing("country (KYC)"))


def add_transaction(report: ET.Element, item: Dict, is_alert: bool) -> None:
    """One evidence row as a goAML-style transaction."""
    block = ET.SubElement(report, "transaction")
    add(block, "transactionnumber", str(item["transaction_id"]))
    add(block, "internal_ref_number", "IBM EdgeID %d" % item["transaction_id"])
    add(block, "transaction_location", missing("branch or channel"))
    what = "alert transaction" if is_alert else "cited by the case note"
    add(block, "transaction_description", "%s, %s" % (item["payment_format"], what))
    add(block, "date_transaction", item["time"].replace(" ", "T") + ":00")
    add(block, "transmode_code", missing("FIU code for %s" % item["payment_format"]))
    add(block, "amount_local", missing("amount in %s" % config.GOAML_LOCAL_CURRENCY))
    add_side(block, "from", item)
    add_side(block, "to", item)


def reported_transactions(evidence: Dict, note: Optional[Dict]) -> List[Dict]:
    """The alert transaction, then the cited rows that are in the evidence, by time."""
    rows = evidence_transactions(evidence)
    alert = rows[0]
    cited = set(note["cited_txn_ids"]) if note else set()
    others = [t for t in rows[1:] if t["transaction_id"] in cited]
    return [alert] + sorted(others, key=lambda t: (t["time"], t["transaction_id"]))


def reason_text(alert: Dict, note: Optional[Dict], escalation: Dict) -> str:
    """Case note summary, red flags and the analyst's comment, within the limit."""
    opening = "Alert %s: ranked %d by the transaction-monitoring model (score %.3f)."
    parts = [opening % (alert["alert_id"], int(alert["rank"]), float(alert["score"]))]
    if note:
        parts.append("Case note (fact-checked against the evidence): " + note["reason"])
        parts += ["Red flag: " + flag for flag in note["red_flags"]]
    parts.append(
        "Analyst comment: " + (escalation.get("comment") or missing("comment"))
    )
    text = " ".join(parts)
    limit = config.GOAML_REASON_MAX_CHARS
    return text if len(text) <= limit else text[: limit - 1] + "…"


def add_header(report: ET.Element, alert: Dict, submitted: str) -> None:
    """Reporting entity, report type, reference, date and reporting person."""
    add(report, "draft_notice", config.GOAML_DRAFT_NOTICE)
    add(report, "rentity_id", missing("reporting entity id (goAML registration)"))
    add(report, "rentity_branch", missing("reporting branch"))
    add(report, "submission_code", "E")
    add(report, "report_code", "STR")
    add(report, "entity_reference", alert["alert_id"])
    add(report, "submission_date", submitted)
    add(report, "currency_code_local", config.GOAML_LOCAL_CURRENCY)
    person = ET.SubElement(report, "reporting_person")
    for field in ("first_name", "last_name", "email"):
        add(person, field, missing("compliance officer %s" % field.replace("_", " ")))
    add(person, "occupation", "Compliance officer")
    location = ET.SubElement(report, "location")
    for field in ("address_type", "address", "city", "country_code"):
        add(location, field, missing("reporting entity %s" % field.replace("_", " ")))


def add_parties(report: ET.Element, note: Optional[Dict], evidence: Dict) -> None:
    """The note's key accounts (or the alert's sender) as report parties."""
    alert = evidence["alert_transaction"]
    accounts = (note or {}).get("key_accounts") or [alert["from_account"]]
    parties = ET.SubElement(report, "report_parties")
    for account in accounts:
        party = ET.SubElement(parties, "report_party")
        add_account(party, "account", account, alert["payment_currency"])
        add(party, "significance", missing("significance (analyst)"))
        add(party, "reason", "Named as a key account in the case note")


def draft_report(
    alert: Dict,
    evidence: Dict,
    record: Optional[Dict],
    escalation: Dict,
    submitted: str,
) -> ET.Element:
    """The whole draft as an element tree (only for an escalated alert)."""
    if escalation.get("decision") != "escalate":
        raise ValueError("A goAML draft is made only for an escalated alert")
    note = (record or {}).get("note")
    report = ET.Element("report")
    add_header(report, alert, submitted)
    add(report, "reason", reason_text(alert, note, escalation))
    add(report, "action", "Escalated by the analyst; draft to be completed and filed "
                          "by the compliance officer.")  # fmt: skip
    alert_id = evidence["alert_transaction"]["transaction_id"]
    for item in reported_transactions(evidence, note):
        add_transaction(report, item, item["transaction_id"] == alert_id)
    add_parties(report, note, evidence)
    indicators = ET.SubElement(report, "report_indicators")
    typology = escalation.get("typology_confirmed")
    if typology:
        add(indicators, "indicator", "Typology %s (analyst confirmed; map to the FIU "
                                     "indicator code)" % typology)  # fmt: skip
    return report


def draft_xml(
    alert: Dict,
    evidence: Dict,
    record: Optional[Dict],
    escalation: Dict,
    submitted: str,
) -> bytes:
    """The draft as UTF-8 XML, with the notice also as a comment on top."""
    report = draft_report(alert, evidence, record, escalation, submitted)
    ET.indent(report)
    body = ET.tostring(report, encoding="unicode")
    header = '<?xml version="1.0" encoding="UTF-8"?>\n<!-- %s -->\n' % (
        config.GOAML_DRAFT_NOTICE
    )
    return (header + body + "\n").encode("utf-8")
