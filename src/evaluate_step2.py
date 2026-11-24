"""Score the case notes against ground truth (evaluation only).

Example:
    report = evaluate_notes(records, truth_table)   # dict written to eval_step2.json
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src import config


def _ratio(top: float, bottom: float) -> float:
    return round(top / bottom, 4) if bottom else 0.0


def decision_scores(valid: pd.DataFrame) -> Dict:
    """Precision and recall of 'escalate' against is_laundering."""
    escalated = valid["decision"] == "escalate"
    real = valid["is_laundering"] == 1
    hits = int((escalated & real).sum())
    return {
        "escalated": int(escalated.sum()),
        "laundering_alerts": int(real.sum()),
        "true_escalations": hits,
        "precision": _ratio(hits, escalated.sum()),
        "recall": _ratio(hits, real.sum()),
    }


def typology_scores(valid: pd.DataFrame) -> Dict:
    """Typology accuracy on alerts whose IBM pattern type is known."""
    mapped = valid[valid["pattern_type"].notna()]
    correct = int((mapped["typology"] == mapped["pattern_type"]).sum())
    return {"mapped_alerts": len(mapped), "correct": correct,
            "accuracy": _ratio(correct, len(mapped))}  # fmt: skip


def evaluate_notes(records: List[Dict], truth: pd.DataFrame, version: str) -> Dict:
    """Validity, decisions, typology, invented facts, cost and latency."""
    frame = pd.DataFrame(records).merge(truth, on="alert_id", how="left")
    valid = frame[frame["note"].notna()].copy()
    for column in ("decision", "typology"):
        valid[column] = valid["note"].map(lambda note: note[column])
    claims = int(valid["claims"].sum())
    invented = int(valid["violations"].map(len).sum())
    cost = (
        frame["input_tokens"].sum() * config.LLM_PRICE_INPUT_PER_MILLION
        + frame["output_tokens"].sum() * config.LLM_PRICE_OUTPUT_PER_MILLION
    ) / 1e6
    return {
        "model": config.LLM_MODEL,
        "prompt_version": version,
        "alerts": len(frame),
        "json_valid": len(valid),
        "json_valid_rate": _ratio(len(valid), len(frame)),
        "decision": decision_scores(valid),
        "typology": typology_scores(valid),
        "notes_with_invented_facts": int((valid["violations"].map(len) > 0).sum()),
        "invented_fact_rate_per_note": _ratio(
            (valid["violations"].map(len) > 0).sum(), len(valid)
        ),
        "invented_fact_rate_per_claim": _ratio(invented, claims),
        "cost_usd": round(float(cost), 4),
        "mean_seconds_per_note": round(float(frame["seconds"].mean()), 2),
    }
