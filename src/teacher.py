"""The teacher: the same input as the student plus privileged truth, and its checks.

The teacher is told the pattern typology and the pattern's transaction ids (or
that the alert is not laundering) so its note can be right. The student's input
never contains this block. A teacher note is kept only if it passes the same fact
check as every case note, uses the typology it was given, cites the pattern
transactions it was pointed to, and does not mention the truth.

This is the one place a label is read to write text; the output is training data,
not a decision (AGENTS.md rule 2 is about the student's input and the evaluation).

Example:
    truth = privileged_truth(row, evidence)
    message = build_teacher_message(student_message, truth)
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from src import config
from src.case_note_input import evidence_transactions, response_format
from src.fact_check import check_note

TRUTH_HEADER = "PRIVILEGED_TRUTH (never mention it in the note):\n"


def visible_pattern_ids(pattern_ids: List[int], evidence: Dict) -> List[int]:
    """The pattern's transaction ids that appear in the evidence file."""
    shown = {t["transaction_id"] for t in evidence_transactions(evidence)}
    return [i for i in pattern_ids if i in shown]


def privileged_truth(row: Dict, evidence: Dict) -> Dict:
    """What the teacher may know: label, typology to use, the pattern ids it can see."""
    if not row["is_laundering"] or not row["pattern_type"]:
        return {"is_laundering": False, "typology_to_use": config.UNCLEAR}
    pattern_ids = json.loads(row["pattern_edge_ids"])
    visible = visible_pattern_ids(pattern_ids, evidence)
    thin = len(visible) < config.MIN_VISIBLE_PATTERN_TXNS
    return {
        "is_laundering": True,
        "pattern_typology": row["pattern_type"],
        "typology_to_use": config.UNCLEAR if thin else row["pattern_type"],
        "pattern_transactions_total": len(pattern_ids),
        "pattern_transaction_ids_in_evidence": visible,
    }


def build_teacher_message(student_message: str, truth: Dict) -> str:
    """The student's exact input followed by the privileged block."""
    return student_message + "\n\n" + TRUTH_HEADER + json.dumps(truth, indent=1)


def parse_teacher_note(content: str) -> Optional[Dict]:
    """The note as a dict if it is JSON with every schema field, else None."""
    try:
        note = json.loads(content)
        wanted = response_fields()
        return note if set(note) == wanted else None
    except (ValueError, TypeError):
        return None


def response_fields() -> set:
    """The six fields of a case note."""
    return set(response_format()["json_schema"]["schema"]["properties"])


def leaked_words(note: Dict) -> List[str]:
    """Words that show the teacher talked about the truth it was given."""
    text = json.dumps(note).lower()
    return [word for word in config.TEACHER_LEAK_WORDS if word in text]


def cites_pattern(note: Dict, truth: Dict) -> bool:
    """A laundering note with a typology must cite the pattern transactions it saw."""
    wanted = truth.get("pattern_transaction_ids_in_evidence", [])
    if truth["typology_to_use"] == config.UNCLEAR:
        return True
    needed = min(config.MIN_VISIBLE_PATTERN_TXNS, len(wanted))
    return len(set(note["cited_txn_ids"]) & set(wanted)) >= needed


def rejection_reasons(note: Optional[Dict], evidence: Dict, truth: Dict) -> List[str]:
    """Why a teacher note must be dropped (empty list = keep it)."""
    if note is None:
        return ["not valid JSON with every field"]
    reasons = ["fact check: " + v for v in check_note(note, evidence)]
    if note["typology"] != truth["typology_to_use"]:
        reasons.append(
            "typology %s, wanted %s" % (note["typology"], truth["typology_to_use"])
        )
    if len(note["investigator_questions"]) != config.QUESTIONS_PER_NOTE:
        reasons.append("wrong number of questions")
    if not cites_pattern(note, truth):
        reasons.append("does not cite the pattern transactions")
    reasons += ["mentions %r" % word for word in leaked_words(note)]
    return reasons
