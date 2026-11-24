"""Loaders for the review UI. They read saved artifacts only (no model, no API).

Example:
    queue = load_queue(config.ARTIFACTS_DIR)        # one row per alert
    bundle = load_bundle(config.ARTIFACTS_DIR, "A0029")
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src import config
from src.case_note_input import evidence_transactions
from src.fact_check import CITED_IDS_IN_TEXT
from src.reasons import display_reasons


def load_notes(folder: Path) -> Dict[str, Dict]:
    """Saved note records by alert id."""
    records = json.loads((folder / config.UI_NOTES_FILE).read_text())
    return {record["alert_id"]: record for record in records}


def test_period_size(folder: Path) -> int:
    """Number of test-period transactions (0 if the summary lacks it)."""
    summary = load_json_if_present(folder / config.SUMMARY_FILE)
    return int(summary.get("test_transactions", 0))


def top_percent_text(rank: int, test_size: int) -> str:
    """ "top 0.0017%": the alert's rank among every test-period score."""
    if not test_size:
        return ""
    return "top %.4f%%" % (100.0 * rank / test_size)


def review_label(needs_review: bool) -> str:
    """Read-only flag text for tables and badges."""
    return "⚠ needs review" if needs_review else "✓ clean"


def reason_line(reason: Dict) -> str:
    """Reason, strength and share; "(lowers the score)" only if it lowered it."""
    line = "%s — %s, %d%% of the top-5 push" % (
        reason["text"],
        reason["strength"],
        reason["share_percent"],
    )
    return line + (" (lowers the score)" if reason["direction"] == "lowers" else "")


def load_queue(folder: Path) -> pd.DataFrame:
    """The alert queue: priority, score, top reason, typology hint, review flag."""
    alerts = pd.read_parquet(folder / config.ALERTS_FILE)
    notes = load_notes(folder)
    size = test_period_size(folder)
    rows = []
    for alert in alerts.to_dict("records"):
        record = notes.get(alert["alert_id"], {})
        note = record.get("note")
        rows.append(
            {
                "alert_id": alert["alert_id"],
                "priority_rank": int(alert["rank"]),
                "score": float(alert["score"]),
                "score_rank": top_percent_text(int(alert["rank"]), size),
                "top_reason": reason_line(display_reasons(alert)[0]),
                "typology_hint": note["typology"] if note else "no note",
                "review": review_label(bool(record.get("needs_review", True))),
            }
        )
    return pd.DataFrame(rows)


def load_bundle(folder: Path, alert_id: str) -> Dict[str, Any]:
    """Everything the detail page shows for one alert (no label, no pattern)."""
    alerts = pd.read_parquet(folder / config.ALERTS_FILE)
    alert = alerts[alerts["alert_id"] == alert_id].iloc[0].to_dict()
    evidence = json.loads(
        (folder / config.EVIDENCE_FOLDER / ("%s.json" % alert_id)).read_text()
    )
    return {
        "alert": alert,
        "reasons": [reason_line(r) for r in display_reasons(alert)],
        "score_rank": top_percent_text(int(alert["rank"]), test_period_size(folder)),
        "evidence": evidence,
        "table": evidence_table(evidence),
        "record": load_notes(folder).get(alert_id),
    }


def evidence_table(evidence: Dict) -> pd.DataFrame:
    """The alert transaction and linked transactions as one table."""
    rows = []
    for item in evidence_transactions(evidence):
        rows.append(
            {
                "transaction_id": item["transaction_id"],
                "hop": item["hop"],
                "time": item["time"],
                "from_account": item["from_account"],
                "to_account": item["to_account"],
                "amount_paid": item["amount_paid"],
                "payment_currency": item["payment_currency"],
                "payment_format": item["payment_format"],
            }
        )
    return pd.DataFrame(rows)


def cited_ids(record: Dict) -> List[int]:
    """Ids the note cites: its list plus ids written in flags and questions."""
    note = record.get("note") if record else None
    if not note:
        return []
    text = " ".join(note["red_flags"] + note["investigator_questions"])
    written = [
        int(n) for g in CITED_IDS_IN_TEXT.findall(text) for n in re.findall(r"\d+", g)
    ]
    return sorted(set(note["cited_txn_ids"]) | set(written))


def load_json_if_present(path: Path) -> Dict:
    """A saved JSON report, or an empty dict when it is missing."""
    return json.loads(path.read_text()) if path.exists() else {}
