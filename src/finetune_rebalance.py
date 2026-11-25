"""Rebalance the train split: keep every typed example, thin out "unclear".

Without this, 60% of the examples teach "unclear" and the student may learn to
say it too often. All examples with a typology stay; "unclear" is cut down to
FINETUNE_UNCLEAR_SHARE of the train split. The two kinds of "unclear" (false alarm,
and laundering whose evidence is too thin) keep their original proportions, and the
cut is seeded. The validation split is never touched.

Example:
    kept_ids = rebalance_ids(meta_rows)     # alert ids to keep in train
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

from src import config

FALSE_ALARM, THIN_LAUNDERING, TYPED = "false_alarm", "thin_laundering", "typed"


def example_kind(row: Dict) -> str:
    """ "typed", "false_alarm" or "thin_laundering" (laundering shown as unclear)."""
    if row["typology_to_use"] != config.UNCLEAR:
        return TYPED
    return THIN_LAUNDERING if row["is_laundering"] else FALSE_ALARM


def unclear_quota(typed: int, kinds: Dict[str, int]) -> Dict[str, int]:
    """How many of each "unclear" kind to keep so unclear is the wanted share."""
    share = config.FINETUNE_UNCLEAR_SHARE
    total = sum(kinds.values())
    wanted = min(total, int(round(typed * share / (1 - share))))
    quota = {k: int(round(wanted * n / total)) for k, n in kinds.items() if n}
    return {k: min(q, kinds[k]) for k, q in quota.items()}


def rebalance_ids(rows: List[Dict]) -> Set[str]:
    """Alert ids kept in train: all typed, a seeded stratified share of unclear."""
    train = [r for r in rows if r["split"] == "train"]
    kept = {r["alert_id"] for r in train if example_kind(r) == TYPED}
    groups = {k: [r["alert_id"] for r in train if example_kind(r) == k]
              for k in (FALSE_ALARM, THIN_LAUNDERING)}  # fmt: skip
    quota = unclear_quota(len(kept), {k: len(v) for k, v in groups.items()})
    generator = np.random.RandomState(config.FINETUNE_REBALANCE_SEED)
    for kind, ids in groups.items():
        chosen = generator.choice(sorted(ids), size=quota.get(kind, 0), replace=False)
        kept.update(str(i) for i in chosen)
    return kept


def example_alert_id(example: Dict) -> str:
    """The alert id inside a chat example's user message."""
    return json.loads(example["messages"][1]["content"])["alert_id"]


def rebalance_files(folder: Path) -> Dict:
    """Keep the full train file as train_full.jsonl and write the rebalanced train."""
    full = folder / "train_full.jsonl"
    train = folder / "train.jsonl"
    if not full.exists():
        full.write_text(train.read_text())
    rows = [json.loads(line) for line in (folder / "meta.jsonl").open()]
    keep = rebalance_ids(rows)
    examples = [json.loads(line) for line in full.open()]
    chosen = [e for e in examples if example_alert_id(e) in keep]
    train.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in chosen))
    by_id = {r["alert_id"]: r for r in rows}
    kinds = Counter(example_kind(by_id[example_alert_id(e)]) for e in chosen)
    return {"train_before": len(examples), "train_after": len(chosen),
            "kinds_after": dict(kinds)}  # fmt: skip
