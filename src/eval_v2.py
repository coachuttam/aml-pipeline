"""The V2 evaluation scheme: one set of numbers for V1, prompt v3 and the student.

Evaluation only (reads ground truth). Laundering alerts are split by whether the
evidence shows the pattern ("visible": at least MIN_VISIBLE_PATTERN_TXNS of the
pattern's transactions are in the evidence). Only then can a note be expected to
name the typology. The five measures:

1. typology accuracy on visible-pattern laundering (higher is better)
2. "unclear" rate on false alarms (higher is better: nothing to flag)
3. "unclear" rate on visible-pattern laundering (lower is better)
4. invented-fact rate: fact-check violations per checked claim (lower is better)
5. JSON validity: notes that parsed with every field (higher is better)

Example:
    report = score_notes(records, truth)   # truth: alert_id, is_laundering,
                                           # pattern_type, pattern_visible
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src import config
from src.evaluate import pattern_coverage
from src.ground_truth import pattern_members, read_patterns


def _rate(count: int, total: int) -> Dict:
    return {
        "count": count,
        "of": total,
        "rate": round(count / total, 4) if total else None,
    }


def score_notes(records: List[Dict], truth: pd.DataFrame) -> Dict:
    """The five measures for one set of notes (records as written by run_step2)."""
    frame = pd.DataFrame(records).merge(truth, on="alert_id", how="left")
    valid = frame[frame["note"].notna()].copy()
    valid["typology"] = valid["note"].map(lambda note: note["typology"])
    visible = valid[(valid["is_laundering"] == 1) & valid["pattern_visible"]]
    false_alarms = valid[valid["is_laundering"] == 0]
    claims = int(valid["claims"].sum())
    invented = int(valid["violations"].map(len).sum())
    return {
        "alerts": len(frame),
        "visible_pattern_laundering_alerts": int(
            ((frame["is_laundering"] == 1) & frame["pattern_visible"]).sum()
        ),
        "false_alarm_alerts": int((frame["is_laundering"] == 0).sum()),
        "typology_accuracy_visible": _rate(
            int((visible["typology"] == visible["pattern_type"]).sum()), len(visible)
        ),
        "unclear_rate_false_alarms": _rate(
            int((false_alarms["typology"] == config.UNCLEAR).sum()), len(false_alarms)
        ),
        "unclear_rate_visible_laundering": _rate(
            int((visible["typology"] == config.UNCLEAR).sum()), len(visible)
        ),
        "invented_facts_per_claim": _rate(invented, claims),
        "json_validity": _rate(len(valid), len(frame)),
    }


def visibility_table(
    folder: Path, truth: pd.DataFrame, raw_csv: Path, patterns_file: Path
) -> pd.DataFrame:
    """alert_id -> pattern_visible for one alert folder (needs the raw CSV)."""
    from src.data import load_dataset

    transactions = load_dataset(raw_csv).transactions
    members = pattern_members(read_patterns(patterns_file), transactions)
    documents = [
        json.loads(path.read_text())
        for path in (folder / config.EVIDENCE_FOLDER).glob("*.json")
    ]
    found = pattern_coverage(documents, truth, members).set_index("alert_id")["found"]
    flags = truth["alert_id"].map(found).fillna(0) >= config.MIN_VISIBLE_PATTERN_TXNS
    return pd.DataFrame({"alert_id": truth["alert_id"], "pattern_visible": flags})


def score_file(folder: Path, notes_name: str, raw_csv: Path, patterns: Path) -> Dict:
    """Score `folder/notes_name` and save artifacts/eval_v2_<name>.json beside it."""
    truth = pd.read_parquet(folder / config.GROUND_TRUTH_FILE)
    flags_path = folder / config.PATTERN_VISIBLE_FILE
    if not flags_path.exists():
        visibility_table(folder, truth, raw_csv, patterns).to_parquet(flags_path)
    truth = truth.merge(pd.read_parquet(flags_path), on="alert_id")
    records = json.loads((folder / notes_name).read_text())
    report = score_notes(records, truth)
    out = folder / (config.EVAL_V2_FILE_TEMPLATE % notes_name.replace(".json", ""))
    out.write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    import sys

    print(json.dumps(
        score_file(config.ARTIFACTS_DIR, sys.argv[1], config.RAW_CSV,
                   config.PATTERNS_FILE), indent=1))  # fmt: skip
