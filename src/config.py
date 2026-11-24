"""Paths, split rules, model settings and artifact names (named constants).

Every number the pipeline relies on is here with the reason beside it. Values copied
from aml-gnn-transfer (the project this pipeline builds on) say so.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CACHE_DIR = PROJECT_ROOT / "cache"  # features, never committed

# The raw LI-Small CSV lives in the aml-gnn-transfer checkout and is read in place
# (never copied, never committed). Override with AML_PIPELINE_RAW_CSV.
GNN_TRANSFER_DIR = Path(
    os.environ.get("AML_GNN_TRANSFER_DIR", Path.home() / "aml-gnn-transfer")
)
RAW_CSV = Path(
    os.environ.get(
        "AML_PIPELINE_RAW_CSV",
        GNN_TRANSFER_DIR / "data" / "Small_LI" / "LI-Small_Trans.csv",
    )
)
# IBM's pattern file for LI-Small, if present, names the typology of each laundering
# transaction. Evaluation only: it never goes to the LLM.
PATTERNS_FILE = RAW_CSV.parent / "LI-Small_Patterns.txt"
# The Phase E record this pipeline must reproduce (read only).
PHASE_E_RECORD = (
    GNN_TRANSFER_DIR
    / "results"
    / "phases"
    / "phaseE"
    / "records"
    / ("lightgbm_scratch_f1.0_s0.json")
)

# Multi-GNN splits LI-Small by whole days into about 60% / 20% / 20%
# (external/Multi-GNN/data_loading.py in aml-gnn-transfer); IBM's formatter counts
# time from 10 seconds before midnight of day 1.
SPLIT_SHARES: Tuple[float, ...] = (0.6, 0.2, 0.2)
FIRST_TIMESTAMP_OFFSET_SECONDS = 10

# LightGBM: the Phase E final ("tuned") settings, chosen on validation only.
# Trained on 100% of the training labels, label-mask seed 0.
MAX_TREES = 2000
EARLY_STOPPING_ROUNDS = 100  # stop when validation PR-AUC stalls this long
LEARNING_RATE = 0.05
LEAVES = 7
MODEL_SEED = 42  # LightGBM random_state; the run seed 0 is added to it
CANDIDATE_WEIGHTS: Tuple[object, ...] = (
    0.5,
    1.0,
    2.0,
    5.0,
)  # picked on validation PR-AUC
RUN_SEED = 0

# Alerts and their evidence.
TOP_ALERTS = 200  # how many of the highest-scored test transactions become alerts
SHAP_REASONS_PER_ALERT = 5
EVIDENCE_HOPS = 2
EVIDENCE_WINDOW_DAYS = 7  # linked transactions within this many days either side
EVIDENCE_MAX_EDGES = 50  # linked transactions kept per alert (nearest in time first)

# Metrics (same definitions as Project 2 and aml-gnn-transfer).
PRECISION_AT_K = 100
ALERT_BUDGETS: Tuple[int, ...] = (10_000, 25_000, 50_000)

# Artifact file names.
ALERTS_FILE = "alerts.parquet"
GROUND_TRUTH_FILE = "ground_truth.parquet"  # evaluation only, kept apart from alerts
EVIDENCE_FOLDER = "evidence"
SUMMARY_FILE = "step1_summary.json"
MODEL_FILE = "lightgbm_li_small.txt"
