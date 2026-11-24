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
EVIDENCE_MAX_EDGES = 50  # linked transactions kept per alert
EVIDENCE_HOP1_EDGES = (
    15  # "flow" strategy: nearest-in-time transactions touching S or R
)
EVIDENCE_FLOW_EDGES = 35  # "flow" strategy: onward and backward along the money trail
# "nearest": the 50 nearest in time within 2 hops. "flow": 15 nearest at hop 1, then
# 35 along the money trail. "nearest" kept: mean pattern coverage 0.491 against 0.486
# for "flow" on the 51 laundering alerts with a known pattern (README).
EVIDENCE_STRATEGY = "nearest"
EVIDENCE_STRATEGIES = ("nearest", "flow")
EVIDENCE_NEAREST_FOLDER = "evidence_nearest"  # the other strategy, for comparison only

# Metrics (same definitions as Project 2 and aml-gnn-transfer).
PRECISION_AT_K = 100
ALERT_BUDGETS: Tuple[int, ...] = (10_000, 25_000, 50_000)

# Artifact file names.
ALERTS_FILE = "alerts.parquet"
GROUND_TRUTH_FILE = "ground_truth.parquet"  # evaluation only, kept apart from alerts
EVIDENCE_FOLDER = "evidence"
SUMMARY_FILE = "step1_summary.json"
MODEL_FILE = "lightgbm_li_small.txt"

# Step 2: LLM case notes (OpenAI). Key: OPENAI_API_KEY in .env (git-ignored).
LLM_MODEL = "gpt-4.1-mini"  # cheap, supports temperature 0 and JSON-schema outputs
LLM_TEMPERATURE = 0.0  # same input -> (nearly) same note, so runs can be compared
LLM_MAX_OUTPUT_TOKENS = (
    3000  # 1200 cut off 10 of 200 long notes (README); this stops runaways
)
LLM_MAX_ATTEMPTS = 3  # retries on a rate limit or network error
LLM_RETRY_SECONDS = 5  # wait between retries
# Dollars per million tokens for LLM_MODEL (OpenAI price list); update with the model.
LLM_PRICE_INPUT_PER_MILLION = 0.40
LLM_PRICE_OUTPUT_PER_MILLION = 1.60
PROMPT_VERSIONS = (
    "v1",
    "v2",
    "v3",
)  # prompts live in src/prompts/case_note_<version>.txt
DEV_ALERTS = 30  # top validation-period alerts used to choose between prompts
DEV_FOLDER = "dev"  # artifacts/dev/: alerts, evidence and ground truth for the dev set
PROMPTS_DIR = PROJECT_ROOT / "src" / "prompts"
LLM_CACHE_DIR = CACHE_DIR / "llm"  # one file per request, never committed
QUESTIONS_PER_NOTE = 3  # investigator questions the note must give
AMOUNT_TOLERANCE = 0.01  # a cited total may differ from the sum by a cent of rounding
NOTES_FILE = "case_notes_step2.json"  # notes plus their fact checks
EVAL_STEP2_FILE = "eval_step2.json"
TYPOLOGIES: Tuple[str, ...] = (
    "FAN_OUT",
    "FAN_IN",
    "GATHER_SCATTER",
    "SCATTER_GATHER",
    "CYCLE",
    "RANDOM",
    "BIPARTITE",
    "STACK",
)
UNCLEAR = "unclear"
