"""Step 1: detect -> evidence -> ground truth, written to artifacts/.

    python -m src.run_step1

Reads the raw LI-Small CSV from the aml-gnn-transfer checkout (see config.py),
trains the Phase E LightGBM detector on the training period (validation picks the
class weight and threshold), scores the test period once, keeps the top 200
alerts with SHAP reason codes, builds each alert's evidence, and writes the
ground truth (evaluation only) to its own file.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src import config
from src.data import Dataset, load_dataset
from src.detect import build_alerts, write_alerts
from src.evidence import EvidenceIndex, build_evidence, write_evidence
from src.features import FEATURE_NAMES, load_or_build
from src.ground_truth import (
    ground_truth_table,
    laundering_share,
    read_patterns,
    write_ground_truth,
)
from src.metrics import score_period
from src.model import Detector, train_detector
from src.reproduce import compare_with_phase_e


def score_test_period(detector: Detector, features: np.ndarray, dataset: Dataset):
    """Test scores for the whole test period at the validation threshold, once."""
    split, truth = dataset.split, dataset.truth
    scores = detector.model.predict_proba(features[split.test])[:, 1]
    return scores, score_period(truth[split.test], scores, detector.threshold)


def write_evidence_files(alerts: pd.DataFrame, dataset: Dataset, folder: Path) -> int:
    """One evidence JSON per alert. Returns how many were written."""
    index = EvidenceIndex(dataset.transactions)
    for alert_id, row in zip(alerts["alert_id"], alerts["row"]):
        write_evidence(build_evidence(index, int(row), alert_id), folder)
    return len(alerts)


def summarise(
    detector: Detector, metrics: Any, truth_table: pd.DataFrame, alerts: pd.DataFrame,
    check: Dict[str, Any], patterns_found: bool, seconds: float,
) -> Dict[str, Any]:  # fmt: skip
    """The numbers to report after a run."""
    return {
        "alerts": int(len(alerts)),
        "alerts_above_validation_threshold": int(alerts["above_threshold"].sum()),
        "laundering_share_of_alerts": laundering_share(truth_table),
        "laundering_in_alerts": int(truth_table["is_laundering"].sum()),
        "pattern_types_available": patterns_found,
        "pattern_counts": truth_table["pattern_type"].value_counts().to_dict(),
        "class_weight": detector.class_weight,
        "threshold": detector.threshold,
        "validation_pr_auc_by_weight": detector.validation_pr_auc_by_weight,
        "test_metrics": metrics.to_dict(),
        "matches_phase_e_record": check,
        "seconds": round(seconds, 1),
    }


def run(output_dir: Path, raw_csv: Path, rebuild: bool = False) -> Dict[str, Any]:
    """The whole of step 1. Returns the summary that is also saved as JSON."""
    start = time.time()
    dataset = load_dataset(raw_csv)
    features = load_or_build(dataset.transactions, config.CACHE_DIR, raw_csv, rebuild)
    detector = train_detector(features, dataset.truth, dataset.split, FEATURE_NAMES)
    scores, metrics = score_test_period(detector, features, dataset)
    alerts = build_alerts(
        dataset.transactions,
        features,
        FEATURE_NAMES,
        scores,
        dataset.split.test,
        detector,
    )
    write_alerts(alerts, output_dir)
    write_evidence_files(alerts, dataset, output_dir / config.EVIDENCE_FOLDER)
    patterns = read_patterns(config.PATTERNS_FILE)
    truth_table = ground_truth_table(
        alerts, dataset.transactions, dataset.truth, patterns
    )
    write_ground_truth(truth_table, output_dir)
    check = compare_with_phase_e(metrics.to_dict(), config.PHASE_E_RECORD)
    summary = summarise(
        detector,
        metrics,
        truth_table,
        alerts,
        check,
        bool(patterns),
        time.time() - start,
    )
    (output_dir / config.SUMMARY_FILE).write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: Optional[list] = None) -> None:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-csv", type=Path, default=config.RAW_CSV)
    parser.add_argument("--output-dir", type=Path, default=config.ARTIFACTS_DIR)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the features")
    arguments = parser.parse_args(argv)
    summary = run(arguments.output_dir, arguments.raw_csv, arguments.rebuild)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
