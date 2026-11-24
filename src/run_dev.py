"""Build the dev set: the top validation-period alerts, used to choose a prompt.

    python -m src.run_dev

Same detector, same SHAP reasons, same evidence rules as the test alerts, but taken
from the validation period so the test period is never used to choose (AGENTS.md
rule 1). Written to artifacts/dev/ (alerts, evidence/, ground truth).
"""

from __future__ import annotations

from pathlib import Path

from src import config
from src.data import load_dataset
from src.detect import build_alerts, write_alerts
from src.features import FEATURE_NAMES, load_or_build
from src.ground_truth import ground_truth_table, read_patterns, write_ground_truth
from src.model import train_detector
from src.run_step1 import write_evidence_files


def build_dev_set(output_dir: Path, raw_csv: Path = config.RAW_CSV) -> Path:
    """Write the dev alerts, their evidence and ground truth; returns the folder."""
    folder = output_dir / config.DEV_FOLDER
    dataset = load_dataset(raw_csv)
    features = load_or_build(dataset.transactions, config.CACHE_DIR, raw_csv, False)
    detector = train_detector(features, dataset.truth, dataset.split, FEATURE_NAMES)
    window = dataset.split.validation
    scores = detector.model.predict_proba(features[window])[:, 1]
    alerts = build_alerts(
        dataset.transactions, features, FEATURE_NAMES, scores, window, detector,
        config.DEV_ALERTS,
    )  # fmt: skip
    write_alerts(alerts, folder)
    write_evidence_files(
        alerts, dataset, folder / config.EVIDENCE_FOLDER, config.EVIDENCE_STRATEGY
    )
    patterns = read_patterns(config.PATTERNS_FILE)
    table = ground_truth_table(alerts, dataset.transactions, dataset.truth, patterns)
    write_ground_truth(table, folder)
    return folder


if __name__ == "__main__":
    print(build_dev_set(config.ARTIFACTS_DIR))
