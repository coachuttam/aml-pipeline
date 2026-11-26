"""Score the student's raw answers: same fact check and eval_v2 as V1 and v3.

    python -m src.score_student --set dev  --raw data/finetune/dev_raw_<run>.jsonl
    python -m src.score_student --set test --raw data/finetune/test_raw_<run>.jsonl
    python -m src.score_student --set val  --raw data/finetune/val_raw_<run>.jsonl

Each answer is parsed (every field must be there), fact-checked against its own
evidence file, and written as `notes_student.json` in the alert folder
(artifacts/dev_finetune/, artifacts/, or data/finetune/val_set/ for the HI-Small
validation set), in the same record format as run_step2. Then the
V2 evaluation scheme (src/eval_v2.py) is written beside it
(eval_v2_notes_student.json). Runs on a CPU; the test set is scored once, from the
file the GPU made.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src import config
from src.case_note_input import evidence_transactions
from src.eval_v2 import score_file
from src.fact_check import check_note, count_claims
from src.teacher import parse_teacher_note
from src.totals import totals_by_currency

NOTES_NAME = "notes_student.json"


def student_record(alert: Dict, evidence: Dict, raw: Dict) -> Dict:
    """One scored record, like run_step2's: note (or None), violations, tokens."""
    note = None if raw.get("truncated") else parse_teacher_note(raw["text"])
    violations: List[str] = []
    if note is not None:
        try:
            violations = check_note(note, evidence)
            by_id = {t["transaction_id"]: t for t in evidence_transactions(evidence)}
            note["totals_by_currency"] = totals_by_currency(
                note["cited_txn_ids"], by_id
            )
        except (KeyError, TypeError):  # right field names, wrong value types
            note = None
    return {
        "alert_id": alert["alert_id"], "priority_rank": int(alert["rank"]),
        "score": float(alert["score"]), "note": note,
        "needs_review": bool(violations) or note is None, "violations": violations,
        "claims": count_claims(note) if note else 0,
        "input_tokens": raw["input_tokens"], "output_tokens": raw["output_tokens"],
        "seconds": 0.0,
    }  # fmt: skip


def score_set(folder: Path, raw_path: Path) -> Dict:
    """Records, step-2 report and V2 report for one alert folder."""
    alerts = pd.read_parquet(folder / config.ALERTS_FILE)
    raw = {r["alert_id"]: r for r in map(json.loads, raw_path.read_text().splitlines())}
    records = []
    for alert in alerts.to_dict("records"):
        name = "%s.json" % alert["alert_id"]
        evidence = json.loads((folder / config.EVIDENCE_FOLDER / name).read_text())
        missing = {"text": "", "input_tokens": 0, "output_tokens": 0}
        records.append(
            student_record(alert, evidence, raw.get(alert["alert_id"], missing))
        )
    (folder / NOTES_NAME).write_text(json.dumps(records, indent=1))
    report = score_file(folder, NOTES_NAME, config.RAW_CSV, config.PATTERNS_FILE)
    report["notes_needing_review"] = sum(r["needs_review"] for r in records)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", choices=["dev", "test", "val"], required=True)
    parser.add_argument("--raw", type=Path, required=True)
    arguments = parser.parse_args()
    folder = config.ARTIFACTS_DIR
    if arguments.set == "dev":
        folder = folder / config.FINETUNE_DEV_FOLDER
    if arguments.set == "val":
        folder = config.FINETUNE_DIR / config.FINETUNE_VAL_FOLDER
    print(json.dumps(score_set(folder, arguments.raw), indent=1))


if __name__ == "__main__":
    main()
