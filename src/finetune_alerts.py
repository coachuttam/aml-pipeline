"""Build the HI-Small "alerts" the case-note model is trained on.

Same detector as V1 (LightGBM, same features and settings, trained on the HI-Small
training period), but the top-scored transactions of the HI-Small validation and
test periods are the alerts, so their scores and SHAP reasons look like real,
unseen ones. Evidence and reason codes come from the existing code. LI-Small is
never read here.

Output in `data/finetune/candidates/` (git-ignored):
    alerts.parquet      what the student sees (no label), evidence/<id>.json
    ground_truth.parquet label, pattern type, attempt, split, pattern transaction ids

Example:
    build_candidates(config.FINETUNE_DIR / "candidates")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Set

import pandas as pd

from src import config
from src.data import load_dataset
from src.detect import build_alerts, write_alerts
from src.features import FEATURE_NAMES, load_or_build
from src.finetune_select import assign_splits, select_alerts
from src.ground_truth import (
    ground_truth_table,
    pattern_members,
    read_patterns,
    write_ground_truth,
)
from src.model import train_detector
from src.run_step1 import write_evidence_files

TRUTH_COLUMNS = ["is_laundering", "pattern_type", "attempt_id"]


def renumber(table: pd.DataFrame) -> Dict[str, str]:
    """New ids A0001.. by score order of the chosen alerts (old id -> new id)."""
    return {
        old: "A%04d" % (position + 1) for position, old in enumerate(table["alert_id"])
    }


def pattern_ids_json(table: pd.DataFrame, members: Dict[int, Set[int]]) -> pd.Series:
    """Edge ids of the whole pattern attempt of each laundering alert, as JSON."""
    return table["attempt_id"].map(
        lambda attempt: (
            json.dumps(sorted(members.get(int(attempt), [])))
            if pd.notna(attempt)
            else "[]"
        )
    )


def choose_candidates(alerts: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Join, select the mix, split by pattern; one row per chosen alert."""
    joined = alerts.merge(truth[["alert_id"] + TRUTH_COLUMNS], on="alert_id")
    return assign_splits(select_alerts(joined))


def build_candidates(folder: Path, raw_csv: Path = config.HI_RAW_CSV) -> Path:
    """Detector, alerts, evidence and truth for the HI-Small training set."""
    dataset = load_dataset(raw_csv)
    features = load_or_build(
        dataset.transactions, config.HI_FEATURE_CACHE, raw_csv, False
    )
    detector = train_detector(features, dataset.truth, dataset.split, FEATURE_NAMES)
    window = slice(dataset.split.validation.start, dataset.split.total)
    scores = detector.model.predict_proba(features[window])[:, 1]
    pool = build_alerts(
        dataset.transactions, features, FEATURE_NAMES, scores, window, detector,
        config.FINETUNE_POOL,
    )  # fmt: skip
    patterns = read_patterns(config.HI_PATTERNS_FILE)
    truth = ground_truth_table(pool, dataset.transactions, dataset.truth, patterns)
    chosen = choose_candidates(pool, truth)
    ids = renumber(chosen)
    chosen["alert_id"] = chosen["alert_id"].map(ids)
    chosen["rank"] = range(1, len(chosen) + 1)
    members = pattern_members(patterns, dataset.transactions)
    truth_out = chosen[["alert_id", "edge_id"] + TRUTH_COLUMNS + ["split"]].copy()
    truth_out["pattern_edge_ids"] = pattern_ids_json(chosen, members)
    alerts_out = chosen.drop(columns=TRUTH_COLUMNS + ["split"])
    write_alerts(alerts_out, folder)
    write_ground_truth(truth_out, folder)
    write_evidence_files(
        alerts_out, dataset, folder / config.EVIDENCE_FOLDER, config.EVIDENCE_STRATEGY
    )
    return folder
