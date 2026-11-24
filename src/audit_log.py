"""The decision log: one JSON line per analyst decision, only ever appended to.

Stored in artifacts/audit_log.jsonl (git-ignored). Each line has the time, alert id,
decision (escalate or close), the analyst's comment and the note version shown.

Example:
    record_decision("A0029", "close", "Looks like a payroll run", "v3")
    read_decisions()[-1]["decision"]   # "close"
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src import config


def default_path() -> Path:
    """Where the log lives unless a test passes another path."""
    return config.ARTIFACTS_DIR / config.AUDIT_LOG_FILE


def record_decision(
    alert_id: str,
    decision: str,
    comment: str,
    note_version: str,
    path: Optional[Path] = None,
) -> Dict[str, str]:
    """Append one decision; a decision other than escalate/close is refused."""
    if decision not in config.DECISIONS:
        raise ValueError("decision must be one of %s" % (config.DECISIONS,))
    event = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "alert_id": alert_id,
        "decision": decision,
        "comment": comment.strip(),
        "note_version": note_version,
    }
    path = path or default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(event) + "\n")
    return event


def read_decisions(path: Optional[Path] = None) -> List[Dict[str, str]]:
    """Every decision, oldest first (empty when none yet)."""
    path = path or default_path()
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def latest_decision_by_alert(path: Optional[Path] = None) -> Dict[str, str]:
    """The most recent decision for each alert id."""
    return {event["alert_id"]: event["decision"] for event in read_decisions(path)}
