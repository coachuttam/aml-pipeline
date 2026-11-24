"""Evaluation only: how much of the true laundering pattern did the evidence find?

Pattern coverage, for a laundering alert whose pattern is known, is the share of
that pattern's transactions that appear in the alert's evidence (the alert
transaction counts as found). It uses the patterns file, so it is an evaluation
number: nothing here feeds the LLM or the choice of alerts (AGENTS.md rule 2).

Example:
    table = pattern_coverage(documents, truth_table, members)
    mean_coverage(table)    # 0.31 means 31% of pattern transactions were found
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Set

import pandas as pd


def evidence_ids(document: Dict[str, Any]) -> Set[int]:
    """Transaction ids in one evidence document, the alert transaction included."""
    found = {int(document["alert_transaction"]["transaction_id"])}
    found.update(int(t["transaction_id"]) for t in document["linked_transactions"])
    return found


def pattern_coverage(
    documents: Iterable[Dict[str, Any]],
    truth_table: pd.DataFrame,
    members: Dict[int, Set[int]],
) -> pd.DataFrame:
    """One row per laundering alert with a known pattern."""
    by_alert = {document["alert_id"]: document for document in documents}
    rows = []
    for _, alert in truth_table.iterrows():
        attempt = alert["attempt_id"]
        if (
            alert["is_laundering"] != 1
            or pd.isna(attempt)
            or int(attempt) not in members
        ):
            continue
        pattern = members[int(attempt)]
        found = evidence_ids(by_alert[alert["alert_id"]]) & pattern
        rows.append(
            {
                "alert_id": alert["alert_id"],
                "pattern_type": alert["pattern_type"],
                "attempt_id": int(attempt),
                "pattern_size": len(pattern),
                "found": len(found),
                "coverage": len(found) / len(pattern),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "alert_id",
            "pattern_type",
            "attempt_id",
            "pattern_size",
            "found",
            "coverage",
        ],
    )


def mean_coverage(table: pd.DataFrame) -> float:
    """Average pattern coverage over the laundering alerts in `table` (0 if none)."""
    return float(table["coverage"].mean()) if len(table) else 0.0


def coverage_summary(table: pd.DataFrame) -> Dict[str, Any]:
    """Headline numbers for one strategy."""
    return {
        "alerts_with_known_pattern": int(len(table)),
        "mean_coverage": mean_coverage(table),
        "median_coverage": float(table["coverage"].median()) if len(table) else 0.0,
        "alerts_with_full_coverage": int((table["coverage"] >= 1.0).sum()),
        "mean_pattern_size": float(table["pattern_size"].mean()) if len(table) else 0.0,
    }
