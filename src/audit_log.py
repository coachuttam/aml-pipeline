"""The decision log: one JSON line per analyst decision, only ever appended to.

Stored in artifacts/audit_log.jsonl (git-ignored). A decision line has the time,
alert id, decision (escalate or close), the analyst's comment, the note version shown
and the typology the analyst confirmed (empty if none). A goAML draft line records
that a draft was made for an escalated alert: event "goaml_draft", how many
placeholders it still has and whether the typology went in.

Clearing the demo log (command line only, never from the UI) moves the file to
artifacts/audit_log_archive/ with a time stamp, so nothing is lost:

    python -m src.audit_log clear          # says what it would do
    python -m src.audit_log clear --yes    # does it

Example:
    record_decision("A0029", "close", "Looks like a payroll run", "v3")
    read_decisions()[-1]["decision"]   # "close"
"""

from __future__ import annotations

import argparse
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
    typology_confirmed: str = "",
) -> Dict[str, str]:
    """Append one decision; a decision other than escalate/close is refused.

    `typology_confirmed` is the suggested typology the analyst confirmed, if any.
    """
    if decision not in config.DECISIONS:
        raise ValueError("decision must be one of %s" % (config.DECISIONS,))
    event = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "alert_id": alert_id,
        "decision": decision,
        "comment": comment.strip(),
        "note_version": note_version,
        "typology_confirmed": typology_confirmed,
    }
    return append(event, path)


def append(event: Dict, path: Optional[Path] = None) -> Dict:
    """Add one line to the log (the only way the log is written)."""
    path = path or default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(event) + "\n")
    return event


def record_draft(
    alert_id: str, placeholders: int, typology: str, path: Optional[Path] = None
) -> Dict:
    """Log that a goAML-style draft was made (never filed by the pipeline)."""
    return append({
        "time": datetime.now().isoformat(timespec="seconds"), "alert_id": alert_id,
        "event": "goaml_draft", "placeholders_left": placeholders,
        "typology_included": typology,
    }, path)  # fmt: skip


def read_decisions(path: Optional[Path] = None) -> List[Dict[str, str]]:
    """Every decision, oldest first (empty when none yet)."""
    path = path or default_path()
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def latest_decisions(path: Optional[Path] = None) -> Dict[str, Dict]:
    """The most recent decision event for each alert id (draft lines skipped)."""
    return {e["alert_id"]: e for e in read_decisions(path) if e.get("decision")}


def latest_decision_by_alert(path: Optional[Path] = None) -> Dict[str, str]:
    """The most recent decision for each alert id."""
    return {k: e["decision"] for k, e in latest_decisions(path).items()}


def current_escalation(alert_id: str, path: Optional[Path] = None) -> Optional[Dict]:
    """The decision event if the alert's latest decision is escalate, else None."""
    event = latest_decisions(path).get(alert_id)
    return event if event and event["decision"] == "escalate" else None


def clear_log(
    path: Optional[Path] = None, archive: Optional[Path] = None
) -> Optional[Path]:
    """Move the log to the archive folder (time-stamped); None if there is no log."""
    path = path or default_path()
    if not path.exists():
        return None
    archive = archive or path.parent / config.AUDIT_ARCHIVE_FOLDER
    archive.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = archive / ("%s_%s%s" % (path.stem, stamp, path.suffix))
    path.rename(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear the demo audit log.")
    parser.add_argument("command", choices=["clear"])
    parser.add_argument("--yes", action="store_true", help="really clear it")
    arguments = parser.parse_args()
    lines = len(read_decisions())
    if not arguments.yes:
        print("%d lines in %s. Run with --yes to move them to %s/."
              % (lines, default_path(), config.AUDIT_ARCHIVE_FOLDER))  # fmt: skip
        return
    target = clear_log()
    print("Nothing to clear." if target is None else
          "Moved %d lines to %s; the log is empty." % (lines, target))  # fmt: skip


if __name__ == "__main__":
    main()
