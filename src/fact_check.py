"""Check a case note against its own evidence file.

A violation is a cited transaction id, a named account or an amount that is not in
the evidence. Notes are checked, never repaired.

Example:
    violations = check_note(note, evidence)   # [] when every fact is in the evidence
"""

from __future__ import annotations

import re
from typing import Dict, List, Set

from src import config
from src.case_note_input import evidence_transactions
from src.totals import totals_by_currency

CITED_IDS_IN_TEXT = re.compile(r"txns?\s+((?:\d+(?:,\s*|\s+and\s+|\s*))+)")
MONEY_IN_TEXT = re.compile(r"\d[\d,]*\.\d{2}\b")  # "10,917.92": a decimal amount


def _evidence_amounts(transactions: List[Dict]) -> Set[float]:
    amounts = set()
    for item in transactions:
        amounts.add(round(item["amount_paid"], 2))
        amounts.add(round(item["amount_received"], 2))
    return amounts


def _near(value: float, options: Set[float]) -> bool:
    return any(abs(value - option) <= config.AMOUNT_TOLERANCE for option in options)


def _check_ids(note: Dict, by_id: Dict[int, Dict]) -> List[str]:
    return [
        "txn id %s not in evidence" % t for t in note["cited_txn_ids"] if t not in by_id
    ]


def _check_accounts(note: Dict, transactions: List[Dict]) -> List[str]:
    known = {t["from_account"] for t in transactions} | {
        t["to_account"] for t in transactions
    }
    return [
        "account %s not in evidence" % a for a in note["key_accounts"] if a not in known
    ]


def _check_text_ids(note: Dict, by_id: Dict[int, Dict]) -> List[str]:
    """Transaction ids written as "(txn 12, 34)" in the text must exist too."""
    text = " ".join(note["red_flags"] + note["investigator_questions"])
    found = [
        int(n)
        for group in CITED_IDS_IN_TEXT.findall(text)
        for n in re.findall(r"\d+", group)
    ]
    return ["txn id %s in text not in evidence" % t for t in found if t not in by_id]


def _check_text_amounts(
    note: Dict, transactions: List[Dict], by_id: Dict[int, Dict]
) -> List[str]:
    allowed = _evidence_amounts(transactions) | {
        round(v, 2) for v in totals_by_currency(note["cited_txn_ids"], by_id).values()
    }
    text = " ".join(
        note["red_flags"] + note["investigator_questions"] + [note["reason"]]
    )
    found = [float(m.replace(",", "")) for m in MONEY_IN_TEXT.findall(text)]
    return [
        "amount %.2f in text not in evidence" % v
        for v in found
        if not _near(v, allowed)
    ]


def check_note(note: Dict, evidence: Dict) -> List[str]:
    """All violations of one note (empty list = every fact checks out)."""
    transactions = evidence_transactions(evidence)
    by_id = {t["transaction_id"]: t for t in transactions}
    return (
        _check_ids(note, by_id)
        + _check_accounts(note, transactions)
        + _check_text_ids(note, by_id)
        + _check_text_amounts(note, transactions, by_id)
    )


def count_claims(note: Dict) -> int:
    """How many checkable items a note holds (ids and accounts)."""
    return len(note["cited_txn_ids"]) + len(note["key_accounts"])
