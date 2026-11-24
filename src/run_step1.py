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
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src import config
from src.data import Dataset, load_dataset
from src.detect import build_alerts, write_alerts
from src.evaluate import coverage_summary, pattern_coverage
from src.evidence import EvidenceIndex, build_evidence, write_evidence
from src.features import FEATURE_NAMES, load_or_build
from src.ground_truth import (
    ground_truth_table,
    laundering_share,
    pattern_members,
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


def write_evidence_files(
    alerts: pd.DataFrame, dataset: Dataset, folder: Path, strategy: str
) -> List[Dict[str, Any]]:
    """One evidence JSON per alert in `folder`. Returns the documents."""
    index = EvidenceIndex(dataset.transactions)
    documents = []
    for alert_id, row in zip(alerts["alert_id"], alerts["row"]):
        document = build_evidence(index, int(row), alert_id, strategy)
        write_evidence(document, folder)
        documents.append(document)
    return documents


def evidence_for_both_strategies(
    alerts: pd.DataFrame, dataset: Dataset, output_dir: Path
) -> Dict[str, List[Dict[str, Any]]]:
    """The configured strategy goes to evidence/, the other to evidence_<name>/.

    The second folder is only for comparison; both are git-ignored.
    """
    documents = {}
    for strategy in config.EVIDENCE_STRATEGIES:
        chosen = strategy == config.EVIDENCE_STRATEGY
        name = config.EVIDENCE_FOLDER if chosen else "evidence_" + strategy
        documents[strategy] = write_evidence_files(
            alerts, dataset, output_dir / name, strategy
        )
    return documents


def coverage_by_strategy(
    documents: Dict[str, List[Dict[str, Any]]],
    truth_table: pd.DataFrame,
    members: Dict[int, set],
) -> Dict[str, Any]:
    """Pattern coverage of each strategy (evaluation only)."""
    return {
        name: coverage_summary(pattern_coverage(docs, truth_table, members))
        for name, docs in documents.items()
    }


def summarise(
    detector: Detector,
    metrics: Any,
    truth_table: pd.DataFrame,
    alerts: pd.DataFrame,
    check: Dict[str, Any],
    patterns_found: bool,
    coverage: Dict[str, Any],
    seconds: float,
) -> Dict[str, Any]:
    """The numbers to report after a run."""
    known = truth_table["is_laundering"].eq(1) & truth_table["pattern_type"].notna()
    return {
        "alerts": int(len(alerts)),
        "alerts_above_validation_threshold": int(alerts["above_threshold"].sum()),
        "laundering_share_of_alerts": laundering_share(truth_table),
        "laundering_in_alerts": int(truth_table["is_laundering"].sum()),
        "pattern_types_available": patterns_found,
        "laundering_alerts_with_known_pattern": int(known.sum()),
        "pattern_counts": truth_table["pattern_type"].value_counts().to_dict(),
        "evidence_strategy": config.EVIDENCE_STRATEGY,
        "pattern_coverage": coverage,
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
    documents = evidence_for_both_strategies(alerts, dataset, output_dir)
    patterns = read_patterns(config.PATTERNS_FILE)
    truth_table = ground_truth_table(
        alerts, dataset.transactions, dataset.truth, patterns
    )
    write_ground_truth(truth_table, output_dir)
    members = pattern_members(patterns, dataset.transactions)
    coverage = coverage_by_strategy(documents, truth_table, members)
    check = compare_with_phase_e(metrics.to_dict(), config.PHASE_E_RECORD)
    summary = summarise(
        detector,
        metrics,
        truth_table,
        alerts,
        check,
        bool(patterns),
        coverage,
        time.time() - start,
    )
    test_window = dataset.split.test
    summary["test_transactions"] = int(test_window.stop - test_window.start)
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
