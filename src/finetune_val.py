"""A 50-alert HI-Small validation set in the dev format, to tell the two apart.

The student was trained on HI-Small and fails on LI-Small dev. Run on held-out
HI-Small validation alerts (same patterns as nothing in train), it shows whether
the student learned typologies at all (good here, bad on dev: the shift is the
problem) or not (bad on both: the training is). The mix copies dev: 20 laundering
alerts whose pattern is visible, spread over typologies and one per pattern
attempt; 20 false alarms; 10 laundering alerts with thin evidence.

Output (data/finetune/, git-ignored): val_set/ with alerts.parquet, evidence/,
ground_truth.parquet and pattern_visible.parquet (truth for scoring only), and
val_inputs.jsonl (system + user message, no truth).

Example:
    chosen = pick_val_alerts(meta_rows, truth_rows)
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src import config


def round_robin_by_type(rows: List[Dict], count: int, seed: int) -> List[Dict]:
    """Up to `count` rows, typologies in turn, new pattern attempts before repeats."""
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    seen_attempts = set()
    firsts, repeats = [], []
    for row in shuffled:
        (repeats if row["attempt_id"] in seen_attempts else firsts).append(row)
        seen_attempts.add(row["attempt_id"])
    queues: Dict[str, List[Dict]] = {}
    for row in firsts + repeats:
        queues.setdefault(row["pattern_type"], []).append(row)
    picked: List[Dict] = []
    while len(picked) < count and any(queues.values()):
        for name in sorted(queues):
            if queues[name] and len(picked) < count:
                picked.append(queues[name].pop(0))
    return picked


def pick_val_alerts(meta: List[Dict], truth: List[Dict]) -> List[str]:
    """Alert ids of the validation set: visible, false alarms, then thin laundering."""
    attempts = {t["alert_id"]: t.get("attempt_id") for t in truth}
    rows = [
        {**m, "attempt_id": attempts.get(m["alert_id"])}
        for m in meta
        if m["split"] == "validation"
    ]
    visible = [r for r in rows if r["is_laundering"] and r["pattern_visible"]]
    hidden = [r for r in rows if r["is_laundering"] and not r["pattern_visible"]]
    false_alarms = [r for r in rows if not r["is_laundering"]]
    seeded = random.Random(config.VAL_SET_SEED)
    picked = round_robin_by_type(visible, config.VAL_SET_VISIBLE, config.VAL_SET_SEED)
    picked += seeded.sample(
        false_alarms, min(config.VAL_SET_FALSE_ALARMS, len(false_alarms))
    )
    picked += seeded.sample(hidden, min(config.VAL_SET_HIDDEN, len(hidden)))
    return [r["alert_id"] for r in picked]


def write_val_folder(candidates: Path, meta: List[Dict], folder: Path) -> int:
    """Copy the chosen alerts, their evidence and their truth into `folder`."""
    truth = pd.read_parquet(candidates / config.GROUND_TRUTH_FILE)
    chosen = pick_val_alerts(meta, truth.to_dict("records"))
    alerts = pd.read_parquet(candidates / config.ALERTS_FILE)
    alerts = alerts[alerts["alert_id"].isin(chosen)].sort_values("rank")
    (folder / config.EVIDENCE_FOLDER).mkdir(parents=True, exist_ok=True)
    alerts.to_parquet(folder / config.ALERTS_FILE, index=False)
    for alert_id in chosen:
        name = "%s.json" % alert_id
        source = candidates / config.EVIDENCE_FOLDER / name
        shutil.copyfile(source, folder / config.EVIDENCE_FOLDER / name)
    columns = ["alert_id", "edge_id", "is_laundering", "pattern_type", "attempt_id"]
    truth[truth["alert_id"].isin(chosen)][columns].to_parquet(
        folder / config.GROUND_TRUTH_FILE, index=False
    )
    visible = {m["alert_id"]: bool(m["pattern_visible"]) for m in meta}
    flags = pd.DataFrame(
        {"alert_id": chosen, "pattern_visible": [visible[i] for i in chosen]}
    )
    flags.to_parquet(folder / config.PATTERN_VISIBLE_FILE, index=False)
    return len(chosen)


def read_meta(path: Path) -> List[Dict]:
    """The truth rows of meta.jsonl (one per kept example)."""
    return [json.loads(line) for line in path.read_text().splitlines() if line]
