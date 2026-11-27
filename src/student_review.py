"""Where the student goes wrong: a typology confusion table and its invented facts.

Evaluation only: reads ground truth and the student's raw answers. The true class of
an alert is its pattern type when the evidence shows the pattern, "unclear (thin)"
for laundering whose pattern the evidence does not show (the right answer there is
"unclear"), and "false alarm" for the rest.

An invented fact is one fact-check violation (src/fact_check.py). Each one is sorted
by kind (transaction id, account, amount) and by how it went wrong, for example an id
one past a run of consecutive evidence ids, or an evidence amount with its decimal
point moved.

Example:
    table = confusion_table(predicted, truth_rows)   # {true: {predicted: count}}
"""

from __future__ import annotations

import re
from typing import Dict, List, Set

from src import config
from src.case_note_input import evidence_transactions

FALSE_ALARM, THIN, INVALID = "false alarm", "unclear (thin)", "invalid"
NEAR_ID_DISTANCE = 2  # an invented id this close to an evidence id was extrapolated
MIN_SHIFT_DIGITS = 5  # shorter amounts match too many evidence amounts by chance


def true_class(row: Dict) -> str:
    """Pattern type (pattern visible), "unclear (thin)" or "false alarm"."""
    if not row["is_laundering"]:
        return FALSE_ALARM
    return row["pattern_type"] if row["pattern_visible"] else THIN


def confusion_table(predicted: Dict[str, str], truth: List[Dict]) -> Dict:
    """{true class: {predicted typology: count}}; a missing answer is "invalid"."""
    table: Dict[str, Dict[str, int]] = {}
    for row in truth:
        guess = predicted.get(row["alert_id"]) or INVALID
        cells = table.setdefault(true_class(row), {})
        cells[guess] = cells.get(guess, 0) + 1
    return table


def table_markdown(table: Dict) -> str:
    """The confusion table as a markdown table (rows true, columns predicted)."""
    order = list(config.TYPOLOGIES) + [config.UNCLEAR, INVALID]
    columns = [c for c in order if any(c in cells for cells in table.values())]
    rows = [c for c in list(config.TYPOLOGIES) + [THIN, FALSE_ALARM] if c in table]
    lines = ["| true \\ predicted | " + " | ".join(columns) + " | total |",
             "|---" * (len(columns) + 2) + "|"]  # fmt: skip
    for name in rows:
        counts = [table[name].get(c, 0) for c in columns]
        cells = " | ".join(str(n) if n else "." for n in counts)
        lines.append("| %s | %s | %d |" % (name, cells, sum(counts)))
    return "\n".join(lines)


ID_IN_VIOLATION = re.compile(r"txn id (\d+)")
AMOUNT_IN_VIOLATION = re.compile(r"amount ([\d.]+)")


def id_cause(value: int, known: Set[int]) -> str:
    """Why an id is wrong: next to a run of evidence ids, or unrelated."""
    if any(abs(value - k) <= NEAR_ID_DISTANCE for k in known):
        return "id next to a run of consecutive evidence ids (extrapolated)"
    return "id unrelated to the evidence ids"


def digits(value: float) -> str:
    """The significant digits of an amount: 29,547.46 and 295,474.62 -> "2954746"."""
    return ("%.2f" % value).replace(".", "").lstrip("0")


def amount_cause(value: float, amounts: Set[float]) -> str:
    """Why an amount is wrong: an evidence amount with its decimal point moved?"""
    mine = digits(value)
    if len(mine) >= MIN_SHIFT_DIGITS and any(
        digits(a).startswith(mine[:-1]) for a in amounts if a > 0
    ):
        return "evidence amount with the decimal point moved"
    return "amount not derived from the evidence"


def classify_violation(violation: str, evidence: Dict) -> Dict:
    """Kind (transaction id, account, amount) and cause of one violation."""
    transactions = evidence_transactions(evidence)
    known = {t["transaction_id"] for t in transactions}
    if violation.startswith("account"):
        return {"kind": "account", "cause": "account not in evidence"}
    found = ID_IN_VIOLATION.search(violation)
    if found:
        where = "in text" if "in text" in violation else "in cited_txn_ids"
        return {"kind": "transaction id (%s)" % where,
                "cause": id_cause(int(found.group(1)), known)}  # fmt: skip
    amounts = {t["amount_paid"] for t in transactions}
    amounts |= {t["amount_received"] for t in transactions}
    value = float(AMOUNT_IN_VIOLATION.search(violation).group(1))
    return {"kind": "amount", "cause": amount_cause(value, amounts)}


def invented_facts(records: List[Dict], evidence: Dict[str, Dict]) -> List[Dict]:
    """One row per violation: alert, typology said, the violation, kind and cause."""
    rows = []
    for record in records:
        for violation in record["violations"]:
            rows.append({
                "alert_id": record["alert_id"],
                "typology": record["note"]["typology"] if record["note"] else None,
                "violation": violation,
                **classify_violation(violation, evidence[record["alert_id"]]),
            })  # fmt: skip
    return rows


ONE_DECIMAL_AMOUNT = re.compile(r"\b\d{1,3}(?:,\d{3})+\.\d\b(?!\d)")


def unchecked_amounts(note: Dict, evidence: Dict) -> List[str]:
    """Amounts like "19,969,689.1" (one decimal, which the fact check skips) that are
    not an evidence amount."""
    text = " ".join(
        note["red_flags"] + note["investigator_questions"] + [note["reason"]]
    )
    amounts = set()
    for item in evidence_transactions(evidence):
        amounts |= {item["amount_paid"], item["amount_received"]}
    return [
        found for found in ONE_DECIMAL_AMOUNT.findall(text)
        if not any(abs(float(found.replace(",", "")) - a) <= config.AMOUNT_TOLERANCE
                   for a in amounts)
    ]  # fmt: skip
