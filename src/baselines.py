"""Baselines beside the student's score on one alert set (evaluation only).

- always GATHER_SCATTER: the most common visible typology, said for every alert;
- the rule baseline (src/typology_rules.py) with settings fixed from dev
  (RULE_SETTINGS_FROM_DEV), never tuned on the set it is scored on;
- V1 prompt v3 (gpt-4.1-mini) notes for the same alerts, if the folder has them,
  re-checked with today's fact check so invented facts are counted the same way.

Example:
    comparison(config.ARTIFACTS_DIR)   # {"always_gather_scatter": {...}, ...}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src import config
from src.eval_v2 import score_notes
from src.fact_check import check_note
from src.typology_rules import RuleSettings, detect_typology, score_rules


def folder_truth(folder: Path) -> pd.DataFrame:
    """Ground truth with pattern_visible of one alert folder."""
    truth = pd.read_parquet(folder / config.GROUND_TRUTH_FILE)
    flags = pd.read_parquet(folder / config.PATTERN_VISIBLE_FILE)
    return truth.merge(flags, on="alert_id")


def folder_evidence(folder: Path, alert_ids: List[str]) -> Dict[str, Dict]:
    """alert_id -> v1 evidence document."""
    return {
        i: json.loads((folder / config.EVIDENCE_FOLDER / (i + ".json")).read_text())
        for i in alert_ids
    }


def typology_baselines(evidence: Dict[str, Dict], truth: List[Dict]) -> Dict:
    """Rule baseline (settings from dev) and the always-GATHER_SCATTER constant."""
    settings = RuleSettings(*config.RULE_SETTINGS_FROM_DEV)
    rules = {i: detect_typology(e, settings) for i, e in evidence.items()}
    constant = {i: "GATHER_SCATTER" for i in evidence}
    return {
        "rules_settings_from_dev": vars(settings),
        "rules": score_rules(rules, truth),
        "always_gather_scatter": score_rules(constant, truth),
    }


def rechecked_v1(folder: Path, evidence: Dict[str, Dict], truth: pd.DataFrame) -> Dict:
    """V1 prompt-v3 notes scored like the student, violations from today's check."""
    records = json.loads((folder / config.V1_COMPARISON_NOTES).read_text())
    for record in records:
        if record["note"] is not None:
            record["violations"] = check_note(
                record["note"], evidence[record["alert_id"]]
            )
    return score_notes(records, truth)


def comparison(folder: Path) -> Dict:
    """Baselines and, when present, V1 prompt v3 for one alert folder."""
    truth = folder_truth(folder)
    evidence = folder_evidence(folder, list(truth["alert_id"]))
    report = typology_baselines(evidence, truth.to_dict("records"))
    if (folder / config.V1_COMPARISON_NOTES).exists():
        report["v1_prompt_v3"] = rechecked_v1(folder, evidence, truth)
    return report
