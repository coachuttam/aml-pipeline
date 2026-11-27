"""goAML-style STR draft for an escalated alert (UAE first). Never filed automatically.

The draft has goAML's main parts: the reporting entity, report code STR, the reason
(from the case note and the analyst's comment), one `transaction` per cited
evidence row (the alert transaction always first), the note's key accounts as
`report_parties` (plus fan members, below), and `report_indicators`. The typology
goes in only if the analyst confirmed it when escalating.

What the pipeline cannot know (KYC, names, addresses, institution ids, FIU codes,
AED amounts of non-USD rows) is written as "MISSING: <what>"; USD rows get their AED
amount at the dirham's USD peg (3.6725), noted in internal_ref_number. For a
confirmed FAN_IN the cited senders, for a confirmed FAN_OUT the cited receivers, join
the key accounts as report parties. and `missing_fields` lists it for
the UI. Every value that is filled comes from the alert's evidence file, the note or
the analyst. The schema is our own (docs/goaml_style_draft.xsd): a goAML-style draft,
not a certified UAE FIU schema.

Example:
    xml = draft_xml(alert, evidence, record, escalation, "2026-11-27T10:00:00")
    validation_errors(xml)   # [] when it fits the schema
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

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


def side_money(item: Dict, side: str):
    """(currency name, amount) paid (from) or received (to)."""
    if side == "from":
        return item["payment_currency"], item["amount_paid"]
    return item["receiving_currency"], item["amount_received"]


def pegged(currency: str) -> bool:
    """Is this the currency the dirham is pegged to (so its AED amount is known)?"""
    return currency_code(currency) == config.PEGGED_CURRENCY


def add_side(parent: ET.Element, side: str, item: Dict) -> None:
    """t_from or t_to: funds code, original currency and amount, account, country."""
    currency, amount = side_money(item, side)
    block = ET.SubElement(parent, "t_" + side)
    add(block, side + "_funds_code", missing("funds type code (FIU list)"))
    money = ET.SubElement(block, side + "_foreign_currency")
    add(money, "foreign_currency_code", currency_code(currency))
    add(money, "foreign_amount", "%.2f" % amount)
    rate = "%.4f" % config.AED_PER_USD if pegged(currency) else None
    add(money, "foreign_exchange_rate", rate or missing("rate to AED on the day"))
    add_account(block, side + "_account", item[side + "_account"], currency)
    add(block, side + "_country", missing("country (KYC)"))


def amount_in_aed(item: Dict) -> Optional[float]:
    """The AED amount when a side is in USD (paid side first), else None."""
    for side in ("from", "to"):
        currency, amount = side_money(item, side)
        if pegged(currency):
            return amount * config.AED_PER_USD
    return None


def add_transaction(report: ET.Element, item: Dict, is_alert: bool) -> None:
    """One evidence row as a goAML-style transaction."""
    local = amount_in_aed(item)
    reference = "IBM EdgeID %d" % item["transaction_id"]
    if local is not None:
        reference += "; amount_local at the AED peg, %.4f AED per USD" % (
            config.AED_PER_USD
        )
    block = ET.SubElement(report, "transaction")
    add(block, "transactionnumber", str(item["transaction_id"]))
    add(block, "internal_ref_number", reference)
    add(block, "transaction_location", missing("branch or channel"))
    what = "alert transaction" if is_alert else "cited by the case note"
    add(block, "transaction_description", "%s, %s" % (item["payment_format"], what))
    add(block, "date_transaction", item["time"].replace(" ", "T") + ":00")
    add(block, "transmode_code", missing("FIU code for %s" % item["payment_format"]))
    aed = "%.2f" % local if local is not None else None
    add(
        block,
        "amount_local",
        aed or missing("amount in %s" % config.GOAML_LOCAL_CURRENCY),
    )
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


def party_accounts(
    note: Optional[Dict], evidence: Dict, typology: str
) -> List[Tuple[str, str, str]]:
    """(account, currency, reason) per report party.

    The note's key accounts, plus, for a confirmed FAN_IN, the senders of the
    cited transactions, and for a confirmed FAN_OUT, their receivers.
    """
    alert = evidence["alert_transaction"]
    keys = (note or {}).get("key_accounts") or [alert["from_account"]]
    parties = [(a, alert["payment_currency"], "Named as a key account in the case "
                "note") for a in keys]  # fmt: skip
    role = config.GOAML_PARTY_ROLES.get(typology)
    if role is None:
        return parties
    seen = set(keys)
    side = "from" if role == "from_account" else "to"
    for item in reported_transactions(evidence, note):
        account = item[role]
        if account not in seen:
            seen.add(account)
            reason = "%s of cited transaction %d (%s, analyst confirmed)" % (
                "Sender" if side == "from" else "Receiver",
                item["transaction_id"], typology,
            )  # fmt: skip
            parties.append((account, side_money(item, side)[0], reason))
    return parties


def add_parties(
    report: ET.Element, note: Optional[Dict], evidence: Dict, typology: str
) -> None:
    """Report parties: key accounts, and fan members for a confirmed fan typology."""
    block = ET.SubElement(report, "report_parties")
    for account, currency, reason in party_accounts(note, evidence, typology):
        party = ET.SubElement(block, "report_party")
        add_account(party, "account", account, currency)
        add(party, "significance", missing("significance (analyst)"))
        add(party, "reason", reason)


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
    add_parties(report, note, evidence, escalation.get("typology_confirmed") or "")
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
