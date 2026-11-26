"""Build the HI-Small "alerts" the case-note model is trained on.

Same detector as V1 (LightGBM, same features and settings, trained on the HI-Small
training period), but the top-scored transactions of the HI-Small validation and
test periods are the alerts, so their scores and SHAP reasons look like real,
unseen ones. Evidence and reason codes come from the existing code. LI-Small is
never read here.

Output in `data/finetune/candidates/` (git-ignored; the pool is in its pool/ folder):
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
from src.finetune_select import assign_splits, select_alerts, top_feature
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


def dev_top_feature_mix(dev_folder: Path) -> Dict[str, float]:
    """Share of the dev alerts (LI-Small validation period) per top-reason feature.

    Only the reasons are read, never a label or the test alerts.
    """
    alerts = pd.read_parquet(dev_folder / config.ALERTS_FILE)
    features = alerts["reasons_json"].map(top_feature)
    return features.value_counts(normalize=True).to_dict()


def build_pool(folder: Path, raw_csv: Path = config.HI_RAW_CSV) -> Path:
    """Detector, then the top FINETUNE_POOL alerts with reasons and truth, saved.

    The pool is saved once (it takes the detector's training time); choosing the
    candidates from it is quick and can be redone.
    """
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
    members = pattern_members(patterns, dataset.transactions)
    truth["pattern_edge_ids"] = pattern_ids_json(truth, members)
    write_alerts(pool, folder)
    write_ground_truth(truth, folder)
    return folder


def select_candidates(
    pool_folder: Path, folder: Path, dev_folder: Path, raw_csv: Path = config.HI_RAW_CSV
) -> Path:
    """Choose the candidates from the saved pool, split them, write evidence."""
    pool = pd.read_parquet(pool_folder / config.ALERTS_FILE)
    truth = pd.read_parquet(pool_folder / config.GROUND_TRUTH_FILE)
    extra = truth[["alert_id"] + TRUTH_COLUMNS + ["pattern_edge_ids"]]
    chosen = assign_splits(
        select_alerts(pool.merge(extra, on="alert_id"), dev_top_feature_mix(dev_folder))
    )
    chosen["old_id"] = chosen["alert_id"]
    chosen["alert_id"] = chosen["alert_id"].map(renumber(chosen))
    chosen["rank"] = range(1, len(chosen) + 1)
    truth_out = chosen[["alert_id", "edge_id"] + TRUTH_COLUMNS]
    truth_out = truth_out.assign(
        split=chosen["split"], pattern_edge_ids=chosen["pattern_edge_ids"]
    )
    drop = TRUTH_COLUMNS + ["split", "pattern_edge_ids", "old_id"]
    alerts_out = chosen.drop(columns=drop)
    write_alerts(alerts_out, folder)
    write_ground_truth(truth_out, folder)
    dataset = load_dataset(raw_csv)
    write_evidence_files(
        alerts_out, dataset, folder / config.EVIDENCE_FOLDER, config.EVIDENCE_STRATEGY
    )
    return folder
