"""Step 2: write an LLM case note for each alert, fact-check it, score it.

    python -m src.run_step2 --set dev  --prompt v1     # 30 validation alerts
    python -m src.run_step2 --set test --prompt v2     # the 200 test alerts

The notes and the scores go to artifacts/dev/ (dev) or artifacts/ (test).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src import config
from src.case_note_input import build_user_message, evidence_transactions
from src.evaluate_step2 import evaluate_notes
from src.fact_check import check_note, count_claims
from src.llm_client import make_client, request_note
from src.totals import totals_by_currency


def parse_note(content: str, evidence: Dict) -> Optional[Dict]:
    """The note as a dict if the JSON has every field, else None."""
    try:
        note = json.loads(content)
        note["investigator_questions"], note["cited_txn_ids"]  # fields must exist
        check_note(note, evidence)
    except (ValueError, KeyError, TypeError):
        return None
    return note


def note_for_alert(client, alert: Dict, evidence: Dict, version: str) -> Dict:
    """One alert -> note, violations, review flag, usage (one retry on bad JSON)."""
    message = build_user_message(alert, evidence)
    result = request_note(client, message, version)
    note = parse_note(result["content"], evidence)
    if note is None:
        retried = request_note(client, message, version, retry=1)
        retried["seconds"] += result["seconds"]
        retried["input_tokens"] += result["input_tokens"]
        retried["output_tokens"] += result["output_tokens"]
        result, note = retried, parse_note(retried["content"], evidence)
    violations = check_note(note, evidence) if note else []
    if note:
        by_id = {t["transaction_id"]: t for t in evidence_transactions(evidence)}
        note["totals_by_currency"] = totals_by_currency(note["cited_txn_ids"], by_id)
    return {
        "alert_id": alert["alert_id"],
        "priority_rank": int(alert["rank"]),  # priority = detector score order
        "score": float(alert["score"]),
        "note": note,
        "needs_review": bool(violations) or note is None,
        "violations": violations,
        "claims": count_claims(note) if note else 0,
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "seconds": result["seconds"],
    }


def run(folder: Path, version: str) -> Dict:
    """Notes for every alert in `folder`; saves notes and the evaluation there."""
    alerts = pd.read_parquet(folder / config.ALERTS_FILE)
    client = make_client()
    records = []
    for alert in alerts.to_dict("records"):
        name = "%s.json" % alert["alert_id"]
        evidence = json.loads((folder / config.EVIDENCE_FOLDER / name).read_text())
        records.append(note_for_alert(client, alert, evidence, version))
    truth = pd.read_parquet(folder / config.GROUND_TRUTH_FILE)
    report = evaluate_notes(records, truth, version)
    notes_name = "notes_step2_%s.json" % version
    (folder / notes_name).write_text(json.dumps(records, indent=1))
    is_test = folder == config.ARTIFACTS_DIR
    report_name = config.EVAL_STEP2_FILE if is_test else "eval_%s.json" % version
    (folder / report_name).write_text(json.dumps(report, indent=1))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=["dev", "test"], required=True)
    parser.add_argument("--prompt", choices=config.PROMPT_VERSIONS, required=True)
    arguments = parser.parse_args()
    folder = config.ARTIFACTS_DIR
    if arguments.set == "dev":
        folder = folder / config.DEV_FOLDER
    print(json.dumps(run(folder, arguments.prompt), indent=1))


if __name__ == "__main__":
    main()
