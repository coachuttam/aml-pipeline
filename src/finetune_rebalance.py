"""Rebalance the train split toward the evaluation mix of laundering and false alarms.

Every laundering example stays: the typed ones and the laundering whose evidence is
too thin to show its pattern (the teacher answered "unclear"). False alarms are
sampled (seeded) until laundering is FINETUNE_TRAIN_LAUNDERING_SHARE of the split, or
all of them are used if there are too few. The validation split is never touched.

Example:
    kept_ids, shortfall = rebalance_ids(meta_rows)   # alert ids to keep in train
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

from src import config

FALSE_ALARM, THIN_LAUNDERING, TYPED = "false_alarm", "thin_laundering", "typed"


def example_kind(row: Dict) -> str:
    """ "typed", "false_alarm" or "thin_laundering" (laundering shown as unclear)."""
    if row["typology_to_use"] != config.UNCLEAR:
        return TYPED
    return THIN_LAUNDERING if row["is_laundering"] else FALSE_ALARM


def false_alarms_wanted(laundering: int) -> int:
    """False alarms that make laundering the wanted share of the split."""
    share = config.FINETUNE_TRAIN_LAUNDERING_SHARE
    return int(round(laundering * (1 - share) / share))


def rebalance_ids(rows: List[Dict]) -> Tuple[Set[str], int]:
    """Alert ids kept in train, and how many false alarms were missing (0 if none)."""
    train = [r for r in rows if r["split"] == "train"]
    kept = {r["alert_id"] for r in train if example_kind(r) != FALSE_ALARM}
    false_alarms = sorted(
        r["alert_id"] for r in train if example_kind(r) == FALSE_ALARM
    )
    wanted = false_alarms_wanted(len(kept))
    generator = np.random.RandomState(config.FINETUNE_REBALANCE_SEED)
    take = min(wanted, len(false_alarms))
    kept.update(str(i) for i in generator.choice(false_alarms, take, replace=False))
    return kept, wanted - take


def example_alert_id(example: Dict) -> str:
    """The alert id of a chat example (top-level, or inside the first user message)."""
    if "alert_id" in example:
        return example["alert_id"]
    return json.loads(example["messages"][1]["content"])["alert_id"]


def rebalance_files(folder: Path) -> Dict:
    """Keep the full train file as train_full.jsonl and write the rebalanced train."""
    full = folder / "train_full.jsonl"
    train = folder / "train.jsonl"
    if not full.exists():
        full.write_text(train.read_text())
    rows = [json.loads(line) for line in (folder / "meta.jsonl").open()]
    keep, shortfall = rebalance_ids(rows)
    examples = [json.loads(line) for line in full.open()]
    chosen = [e for e in examples if example_alert_id(e) in keep]
    train.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in chosen))
    by_id = {r["alert_id"]: r for r in rows}
    kinds = Counter(example_kind(by_id[example_alert_id(e)]) for e in chosen)
    laundering = kinds[TYPED] + kinds[THIN_LAUNDERING]
    return {
        "train_before": len(examples), "train_after": len(chosen),
        "kinds_after": dict(kinds), "false_alarms_missing": shortfall,
        "laundering_share": round(laundering / max(len(chosen), 1), 3),
        "unclear_share": round((len(chosen) - kinds[TYPED]) / max(len(chosen), 1), 3),
    }  # fmt: skip
