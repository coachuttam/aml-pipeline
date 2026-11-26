"""Cue distributions: what the first reason and the score say about the class.

A student can take a shortcut when a cue (the model score, the first reason) predicts
the class in the training data but not in the dev or test data. This module counts
those cues per alert set so the sets can be compared, before and after a fix.

Evaluation of the data only: labels are read for the share of laundering, never fed
to a model.

Example:
    profile = cue_profile(alerts, truth)
    profile["laundering_share"], profile["top_feature_share"]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from src import config
from src.finetune_rebalance import example_alert_id
from src.reasons import display_reasons

SCORE_BINS = [
    ("below 0.5", 0.0, 0.5),
    ("0.5 to 0.9", 0.5, 0.9),
    ("0.9 to 0.9999", 0.9, 0.9999),
    ("saturated (1.0)", 0.9999, 2.0),
]
TOP_FEATURES_SHOWN = 4


def _share(values: pd.Series) -> Dict[str, float]:
    return {str(k): round(float(v), 3) for k, v in values.items()}


def cue_profile(alerts: pd.DataFrame, truth: pd.DataFrame) -> Dict:
    """Score bins, first reason, its strength and the laundering share of each."""
    frame = alerts.merge(truth[["alert_id", "is_laundering"]], on="alert_id")
    reasons = frame["reasons_json"].map(json.loads)
    frame["top_feature"] = reasons.map(lambda r: r[0]["feature"])
    frame["first_strength"] = [
        display_reasons(a)[0]["strength"] for a in frame.to_dict("records")
    ]
    bins = {
        name: frame[(frame["score"] >= low) & (frame["score"] < high)]
        for name, low, high in SCORE_BINS
    }
    shown = frame["top_feature"].value_counts().head(TOP_FEATURES_SHOWN).index
    by_feature = frame[frame["top_feature"].isin(shown)].groupby("top_feature")
    return {
        "alerts": len(frame),
        "laundering_share": round(float(frame["is_laundering"].mean()), 3),
        "score_bin_share": {k: round(len(v) / len(frame), 3) for k, v in bins.items()},
        "laundering_share_by_score_bin": {
            k: round(float(v["is_laundering"].mean()), 3)
            for k, v in bins.items()
            if len(v)
        },
        "top_feature_share": _share(
            frame["top_feature"].value_counts(normalize=True).head(TOP_FEATURES_SHOWN)
        ),
        "laundering_share_by_top_feature": _share(by_feature["is_laundering"].mean()),
        "first_reason_strength_share": _share(
            frame["first_strength"].value_counts(normalize=True)
        ),
    }


def train_subset(candidates: Path, train_file: Path) -> Dict:
    """Cue profile of the alerts that are in a train file (read by alert id)."""
    ids = {example_alert_id(json.loads(line)) for line in train_file.open()}
    alerts = pd.read_parquet(candidates / config.ALERTS_FILE)
    truth = pd.read_parquet(candidates / config.GROUND_TRUTH_FILE)
    return cue_profile(alerts[alerts["alert_id"].isin(ids)], truth)


def folder_profile(folder: Path) -> Dict:
    """Cue profile of an alert folder (artifacts/dev_finetune, artifacts)."""
    alerts = pd.read_parquet(folder / config.ALERTS_FILE)
    return cue_profile(alerts, pd.read_parquet(folder / config.GROUND_TRUTH_FILE))


def build_cue_report() -> Dict:
    """Train (before and after the fix), dev and test. The test is only counted."""
    data = config.FINETUNE_DIR
    return {
        "train_before": train_subset(
            data / "candidates_v1", data / "old_v1" / "train.jsonl"
        ),
        "train_after": train_subset(
            data / config.FINETUNE_CANDIDATES_FOLDER, data / "train.jsonl"
        ),
        "dev": folder_profile(config.ARTIFACTS_DIR / config.FINETUNE_DEV_FOLDER),
        "test": folder_profile(config.ARTIFACTS_DIR),
    }
