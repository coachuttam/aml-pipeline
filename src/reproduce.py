"""Check that this pipeline's detector matches aml-gnn-transfer's Phase E record.

The detector is retrained here (about two minutes) with the same data order, split,
features and settings, so its test scores should equal the stored record's. The
record is read only; a difference is reported, never hidden.

Example:
    check = compare_with_phase_e(test_metrics, config.PHASE_E_RECORD)
    check["matches"]   # True when minority F1 agrees to 3 decimals
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

TOLERANCE = 1e-3  # absolute difference allowed in minority F1 and PR-AUC


def compare_with_phase_e(metrics: Dict[str, Any], record_path: Path) -> Dict[str, Any]:
    """Our test metrics against the Phase E record's (`matches` is False if missing)."""
    if not record_path.exists():
        return {"matches": False, "reason": "record not found: %s" % record_path}
    stored = json.loads(record_path.read_text())["metrics"]["scored"]
    gaps = {
        name: abs(float(metrics[name]) - float(stored[name]))
        for name in ("minority_f1", "pr_auc")
    }
    return {
        "matches": all(gap <= TOLERANCE for gap in gaps.values()),
        "ours": {name: float(metrics[name]) for name in gaps},
        "phase_e": {name: float(stored[name]) for name in gaps},
        "gaps": gaps,
    }
