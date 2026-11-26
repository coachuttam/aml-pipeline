"""Reuse teacher notes that do not depend on the fields the student input lost.

The student input no longer has the model score or the size of each reason's push.
A note written before that change is still correct for the same transaction (same
evidence, same truth) if it never talks about the score, the model or a push. Those
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
OLD_EXAMPLE_FILES = ("train_full.jsonl", "validation.jsonl")


def can_reuse(note: Dict) -> bool:
    """True when the note's text never mentions the score, the model or a push."""
    return not REMOVED_FIELDS.search(json.dumps(note))


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
