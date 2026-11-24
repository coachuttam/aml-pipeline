"""Detect: score the test period, keep the top alerts, explain them with SHAP.

The alerts are the highest-scored test transactions. Each gets the five features
with the largest SHAP value (TreeExplainer, log-odds) as reason codes, written
as "feature = value (+0.84)": a plus pushed the score up, a minus pushed it down.
The alert table holds no label; ground truth is written apart (ground_truth.py).

Example:
    alerts = build_alerts(
        transactions, features, names, test_scores, test_slice, detector
    )
    alerts[["alert_id", "score", "reason_1"]].head()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from src import config
from src.data import account_names

REASON_COLUMNS = ["reason_%d" % (i + 1) for i in range(config.SHAP_REASONS_PER_ALERT)]


def top_positions(scores: np.ndarray, count: int) -> np.ndarray:
    """Positions of the `count` highest scores, best first (ties keep file order)."""
    return np.argsort(-scores, kind="stable")[:count]


def alert_id_for(rank: int) -> str:
    """Stable id by rank, for example A0001."""
    return "A%04d" % rank


def shap_matrix(booster: Any, features: pd.DataFrame) -> np.ndarray:
    """SHAP values from TreeExplainer, one row per alert, in log-odds."""
    import shap

    values = shap.TreeExplainer(booster).shap_values(features)
    if isinstance(values, list):  # older SHAP: [negative class, positive class]
        values = values[-1]
    return np.asarray(values)


def reasons_for(
    values: np.ndarray, features: pd.DataFrame, count: int
) -> List[List[dict]]:
    """For each alert, its `count` strongest features as feature/value/shap dicts."""
    reasons = []
    for alert_values, (_, alert_features) in zip(values, features.iterrows()):
        strongest = np.argsort(-np.abs(alert_values), kind="stable")[:count]
        reasons.append(
            [
                {
                    "feature": str(features.columns[i]),
                    "value": float(alert_features.iloc[i]),
                    "shap": float(alert_values[i]),
                }
                for i in strongest
            ]
        )
    return reasons


def reason_text(reason: dict) -> str:
    """One reason as text: "sender_sent_7d = 82 (+14.03)"."""
    return "%s = %.4g (%+.2f)" % (reason["feature"], reason["value"], reason["shap"])


def transaction_columns(transactions: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """The readable columns of the alert transactions (no label exists here)."""
    chosen = transactions.iloc[rows]
    sender, receiver = account_names(chosen)
    return pd.DataFrame(
        {
            "edge_id": chosen["edge_id"].to_numpy(),
            "row": rows,
            "timestamp": chosen["timestamp"].to_numpy(),
            "from_account": sender.to_numpy(),
            "to_account": receiver.to_numpy(),
            "amount_paid": chosen["amount_paid"].to_numpy(),
            "payment_currency": chosen["payment_currency"].astype(str).to_numpy(),
            "amount_received": chosen["amount_received"].to_numpy(),
            "receiving_currency": chosen["receiving_currency"].astype(str).to_numpy(),
            "payment_format": chosen["payment_format"].astype(str).to_numpy(),
        }
    )


def build_alerts(
    transactions: pd.DataFrame,
    features: np.ndarray,
    names: List[str],
    test_scores: np.ndarray,
    test_slice: slice,
    detector: Any,
    count: Optional[int] = None,
) -> pd.DataFrame:
    """The top test transactions (`count`, default config.TOP_ALERTS) with reasons."""
    count = config.TOP_ALERTS if count is None else count
    best = top_positions(test_scores, count)
    rows = best + test_slice.start
    table = transaction_columns(transactions, rows)
    table.insert(0, "rank", np.arange(1, len(rows) + 1))
    table.insert(0, "alert_id", [alert_id_for(r) for r in table["rank"]])
    table["score"] = test_scores[best]
    table["above_threshold"] = table["score"] >= detector.threshold
    alert_features = pd.DataFrame(features[rows], columns=names)
    values = shap_matrix(detector.model.booster_, alert_features)
    reasons = reasons_for(values, alert_features, config.SHAP_REASONS_PER_ALERT)
    for position, column in enumerate(REASON_COLUMNS):
        table[column] = [reason_text(r[position]) for r in reasons]
    table["reasons_json"] = [json.dumps(r) for r in reasons]
    return table


def write_alerts(table: pd.DataFrame, folder: Path) -> Path:
    """Save the alerts as parquet in `folder`."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / config.ALERTS_FILE
    table.to_parquet(path, index=False)
    return path
