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

MONEY_IN_TEXT = re.compile(r"\d[\d,]*\.\d{2}\b")  # "10,917.92": a decimal amount


def _evidence_amounts(transactions: List[Dict]) -> Set[float]:
    amounts = set()
    for item in transactions:
        amounts.add(round(item["amount_paid"], 2))
        amounts.add(round(item["amount_received"], 2))
    return amounts


def _near(value: float, options: Set[float]) -> bool:
    return any(abs(value - option) <= config.AMOUNT_TOLERANCE for option in options)


def total_cited(note: Dict, by_id: Dict[int, Dict]) -> float:
    """Sum of amount_paid over the cited transactions that exist (computed in code)."""
    return sum(by_id[t]["amount_paid"] for t in note["cited_txn_ids"] if t in by_id)


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


def _check_text_amounts(
    note: Dict, transactions: List[Dict], by_id: Dict[int, Dict]
) -> List[str]:
    allowed = _evidence_amounts(transactions) | {round(total_cited(note, by_id), 2)}
    text = " ".join(note["red_flags"] + [note["reason"]])
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
        + _check_text_amounts(note, transactions, by_id)
    )


def count_claims(note: Dict) -> int:
    """How many checkable items a note holds (ids and accounts)."""
    return len(note["cited_txn_ids"]) + len(note["key_accounts"])
