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
# Evidence v2 ("capped"): dev alerts are hubs, so 50 hop-1 rows left no room for hop 2.
EVIDENCE_V2_STRATEGY = "capped"  # nearest in time, but each hop has its own cap
EVIDENCE_V2_HOP1_EDGES = 35  # at most this many hop-1 rows (touching S or R)
EVIDENCE_V2_HOP2_EDGES = 15  # at most this many hop-2 rows, kept even for a hub
EVIDENCE_V2_FOLDER = "evidence_v2"  # beside evidence/ in an alert folder

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
    "v4",
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

# Step 3: review UI (reads saved artifacts only).
UI_NOTES_FILE = "notes_step2_v3.json"  # the notes the UI shows (prompt v3)
UI_NOTE_VERSION = "v3"  # recorded in the audit log with each decision
AUDIT_LOG_FILE = "audit_log.jsonl"  # in artifacts/, so it stays out of git
DECISIONS = ("escalate", "close")  # what the analyst can choose
CITED_COLOUR = "#fff3b0"  # row highlight for transactions the note cites
ALERT_EDGE_COLOUR = "#d62728"  # the alert transaction in the network graph
GRAPH_HEIGHT_PIXELS = 460  # height of the interactive network next to the table
GRAPH_PADDING = 0.25  # extra axis room (share of the spread) so labels are not clipped
CITED_EDGE_COLOUR = "#ff7f0e"  # transactions the note cites
OTHER_EDGE_COLOUR = "#b0b0b0"  # other evidence transactions
NODE_COLOUR = "#4c78a8"  # account dots

# UI fixes: money, scores and reason strength.
CURRENCY_CODES = {  # IBM currency names -> codes, so totals read "USD 10.00"
    "US Dollar": "USD", "Euro": "EUR", "Yuan": "CNY", "Shekel": "ILS", "Rupee": "INR",
    "Yen": "JPY", "UK Pound": "GBP", "Ruble": "RUB", "Canadian Dollar": "CAD",
    "Australian Dollar": "AUD", "Swiss Franc": "CHF", "Brazil Real": "BRL",
    "Mexican Peso": "MXN", "Saudi Riyal": "SAR", "Bitcoin": "BTC",
}  # fmt: skip
STRONG_REASON_SHARE = (
    0.30  # a reason with at least this share of the top-5 push is "strong"
)
MODERATE_REASON_SHARE = 0.15  # at least this share is "moderate"; below it "weak"

# V2 step 1: fine-tuning set for the case-note model (HI-Small, never LI-Small test).
HI_RAW_CSV = GNN_TRANSFER_DIR / "data" / "Small_HI" / "HI-Small_Trans.csv"
HI_PATTERNS_FILE = HI_RAW_CSV.parent / "HI-Small_Patterns.txt"
HI_FEATURE_CACHE = CACHE_DIR / "features_hi_small"  # apart from the LI-Small cache
FINETUNE_DIR = PROJECT_ROOT / "data" / "finetune"  # git-ignored
FINETUNE_CANDIDATES_FOLDER = (
    "candidates"  # alerts, evidence and truth before the teacher
)
FINETUNE_POOL = 60_000  # top-scored HI validation+test transactions to choose from
FINETUNE_ALERTS = 2_200  # candidates kept; the fact check drops some of them
FINETUNE_EXTRA_FALSE_ALARMS = (
    200  # more train-only false alarms, to reach the laundering share
)
FINETUNE_LAUNDERING_SHARE = 0.6  # about 60% laundering, 40% false alarms
FINETUNE_ALERTS_PER_PATTERN = 8  # cap per pattern attempt, so no pattern dominates
FINETUNE_FALSE_ALARMS_PER_SENDER = 2  # cap per sender, so false alarms differ
FINETUNE_VALIDATION_SHARE = 0.1  # share of patterns (and false alarms) held out
FINETUNE_SPLIT_SEED = 0  # shuffles patterns into train / validation
FINETUNE_SAMPLE_SEED = 0  # picks the first small batch for review
FINETUNE_DEV_ALERTS = 50  # LI-Small validation-period alerts to pick checkpoints
FINETUNE_DEV_FOLDER = "dev_finetune"  # under artifacts/ (the 30-alert dev set is kept)
STUDENT_PROMPT_VERSION = "v4"  # system prompt the student sees, in training and at test
BLIND_PROMPT_VERSIONS = (
    "v4",
)  # versions whose input has no model score and no push sizes
TEACHER_PROMPT_VERSION = "teacher"  # src/prompts/case_note_teacher.txt
TEACHER_WORKERS = 8  # parallel API calls; the cache makes reruns free
MIN_VISIBLE_PATTERN_TXNS = 3  # fewer visible: the evidence is thin, typology "unclear"
TEACHER_LEAK_WORDS = (
    "ground truth", "privileged", "labelled", "labeled", "told that", "confirmed",
)  # fmt: skip
TOKEN_ENCODING = "o200k_base"  # tokenizer of gpt-4.1-mini, for length statistics

# V2 evaluation scheme (same for V1, prompt v3 and the fine-tuned student).
PATTERN_VISIBLE_FILE = "pattern_visible.parquet"  # test alerts, evaluation only
EVAL_V2_FILE_TEMPLATE = "eval_v2_%s.json"  # one report per notes file, in artifacts/

# V2 training set: rebalance, and the Colab student.
FINETUNE_TRAIN_LAUNDERING_SHARE = 0.5  # laundering share of train after rebalancing
FINETUNE_REBALANCE_SEED = 0  # picks which "unclear" examples stay
FINETUNE_TEST_INPUTS = "test_inputs.jsonl"  # the 200 test alerts, inputs only

# V2 diagnosis: HI-Small validation inputs, and the rule-based typology baseline.
FINETUNE_VAL_FOLDER = "val_set"  # under data/finetune/: 50 HI validation alerts
FINETUNE_VAL_INPUTS = "val_inputs.jsonl"  # their inputs only, for the Colab notebook
VAL_SET_VISIBLE = 20  # visible-pattern laundering, about as many as dev's 19
VAL_SET_FALSE_ALARMS = 20  # about as many as dev's 21
VAL_SET_HIDDEN = 10  # laundering whose pattern the evidence does not show, as dev's 10
VAL_SET_SEED = 0  # picks which validation alerts go in
RULE_WINDOW_HOURS = (6, 12, 24, 72, 168)  # time windows the rule baseline tries
RULE_MIN_DEGREES = (2, 3, 4, 5, 6, 8)  # counterparties that make a fan
RULE_HUB_MIN_PAYERS = (1, 2, 3)  # payers that make a gather-scatter hub
QWEN_TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"  # for loss-token counts, if installed

# V2 final retrain (train_v3.jsonl): the student answered "unclear" for nearly all.
FINETUNE_V3_TRAIN = "train_v3.jsonl"  # the Colab notebook trains on this file
FINETUNE_V3_PER_TYPOLOGY = 140  # near the median class (115): caps 231, doubles 76
FINETUNE_V3_UNCLEAR_SHARE = 0.35  # was 63.5%: the student learned to always hedge
FINETUNE_V3_FALSE_ALARM_SHARE = 0.75  # of "unclear"; the rest thin laundering (78/22)
FINETUNE_V3_SEED = 0  # picks capped examples, duplicates and the file order
L4_TOKENS_PER_SECOND = 550  # training speed assumed for estimates (not measured)
VAL_GATE_MIN_TYPOLOGY_CORRECT = 10  # of 20 visible-pattern val alerts, before dev
VAL_GATE_MIN_UNCLEAR_FALSE_ALARMS = 0.5  # share of val false alarms said "unclear"
EVAL_SPEEDUP = 3  # a forward-only validation pass costs about a third of a train step
CHAT_TEMPLATE_TOKENS = 20  # role markers Qwen's chat template adds per example

# Final evaluation of the frozen student (typology2 adapter, v1 evidence inputs).
FROZEN_STUDENT_RUN = "typology2"  # the only run whose test answers may be scored
RULE_SETTINGS_FROM_DEV = (24, 4, 1)  # window h, degree, hub payers: best on dev v1
V1_COMPARISON_NOTES = "notes_step2_v3.json"  # gpt-4.1-mini, prompt v3, same alerts
