"""Reuse teacher notes that do not depend on the fields the student input lost.

The student input is now the evidence only: no model score, no reasons, no counts.
A note written before that change is still correct for the same transaction (same
evidence, same truth) if it never talks about the score, the model, a push or a claim
that only the reason sentences or the "found" count could support. Those
notes are reused instead of paying the teacher again; the rest are regenerated.

Example:
    old = load_old_notes(old_folder, old_candidates)   # {edge_id: note}
    can_reuse(old[edge_id])                            # True if it names no score
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict

import pandas as pd

from src import config

REMOVED_FIELDS = re.compile(r"\bscore|\bmodel\b|\braises\b|\blowers\b|\bpush", re.I)
# Wording that comes from the SHAP reason sentences (counts over the last hour or days,
# new senders, usual amounts, earlier payments) or from the evidence's "found" count:
# none of it can be seen in the 50 evidence transactions.
REASON_CLAIMS = re.compile(
    r"\blast (?:\d+ )?(?:hour|hours|day|days|24 hours|7 days)\b"
    r"|\bwithin the (?:last|past) \w+ (?:hour|day)"
    r"|\b\d+ new \w+|\bnew (?:senders|accounts|partners) in\b"
    r"|\b(?:usual|average|typical|mean) (?:amount|payment|transaction)"
    r"|\btimes the\b|\bso far\b|\b(?:in|out)[- ]?degree\b|\bratio\b"
    r"|first[- ]time|not paid each other|paid each other before"
    r"|\blinked transactions (?:were )?found|\b\d[\d,]* linked transactions",
    re.I,
)
OLD_EXAMPLE_FILES = ("train_full.jsonl", "validation.jsonl")


def relies_on_removed_input(note: Dict) -> bool:
    """True when the note talks about the score, the model, a push or a reason claim."""
    text = json.dumps(note)
    return bool(REMOVED_FIELDS.search(text) or REASON_CLAIMS.search(text))


def can_reuse(note: Dict) -> bool:
    """True when the note can be kept for the evidence-only input."""
    return not relies_on_removed_input(note)


def load_old_notes(old_folder: Path, old_candidates: Path) -> Dict[int, Dict]:
    """Kept teacher notes of the first dataset, by transaction id (edge id)."""
    path = old_candidates / config.GROUND_TRUTH_FILE
    if not path.exists():
        return {}
    edges = pd.read_parquet(path).set_index("alert_id")["edge_id"].to_dict()
    notes: Dict[int, Dict] = {}
    for name in OLD_EXAMPLE_FILES:
        if not (old_folder / name).exists():
            continue
        for line in (old_folder / name).open():
            example = json.loads(line)
            alert_id = json.loads(example["messages"][1]["content"])["alert_id"]
            notes[int(edges[alert_id])] = json.loads(example["messages"][2]["content"])
    return notes
