"""Scores for one run, defined as in Project 2 (aml-ibm-experiment/src/ibm/train.py).

Minority F1, precision and recall are for the laundering class at one score
threshold. The threshold is chosen on the validation period only; the test
period is scored once with that fixed threshold.

Example:
    metrics = score_run(val_truth, val_scores, test_truth, test_scores)
    metrics.minority_f1
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.config import ALERT_BUDGETS, PRECISION_AT_K


@dataclass(frozen=True)
class RunMetrics:
    """How one run did on one period. Budget keys are strings, for JSON."""

    minority_f1: float
    precision: float
    recall: float
    pr_auc: float
    precision_at_100: float
    alerts: int
    threshold: float
    recall_at_alert_budget: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Plain dictionary for the JSON run record."""
        return asdict(self)


def top_scored_positions(scores: np.ndarray, count: int) -> np.ndarray:
    """Positions of the `count` highest scores. Stable, so ties keep file order."""
    return np.argsort(-scores, kind="stable")[:count]


def precision_at_top(
    truth: np.ndarray, scores: np.ndarray, top_count: int = PRECISION_AT_K
) -> float:
    """Share of the `top_count` highest-scored transactions that are laundering."""
    best = top_scored_positions(scores, top_count)
    return float(truth[best].mean()) if len(best) else 0.0


def recall_at_budget(truth: np.ndarray, scores: np.ndarray, budget: int) -> float:
    """Share of all laundering rows found within the top `budget` alerts."""
    positives = int(truth.sum())
    if positives == 0:
        return 0.0
    best = top_scored_positions(scores, budget)
    return int(truth[best].sum()) / positives


def best_f1_threshold(truth: np.ndarray, scores: np.ndarray) -> float:
    """The score cut-off with the best minority-class F1 (use validation data only)."""
    if int(truth.sum()) == 0:
        raise ValueError("Cannot choose a threshold: no laundering rows in this period")
    precision, recall, thresholds = precision_recall_curve(truth, scores)
    # The last precision/recall pair has no threshold, so it is left out.
    denominator = np.maximum(precision[:-1] + recall[:-1], 1e-12)
    f1 = 2 * precision[:-1] * recall[:-1] / denominator
    return float(thresholds[int(np.argmax(f1))])


def score_period(truth: np.ndarray, scores: np.ndarray, threshold: float) -> RunMetrics:
    """All metrics for one period at a fixed threshold."""
    flagged = scores >= threshold
    true_positive = int((flagged & (truth == 1)).sum())
    precision = true_positive / flagged.sum() if flagged.any() else 0.0
    recall = true_positive / max(int((truth == 1).sum()), 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    budgets = {
        str(budget): recall_at_budget(truth, scores, budget) for budget in ALERT_BUDGETS
    }
    return RunMetrics(
        minority_f1=f1,
        precision=float(precision),
        recall=recall,
        pr_auc=float(average_precision_score(truth, scores)),
        precision_at_100=precision_at_top(truth, scores),
        alerts=int(flagged.sum()),
        threshold=float(threshold),
        recall_at_alert_budget=budgets,
    )


def score_run(
    validation_truth: np.ndarray,
    validation_scores: np.ndarray,
    test_truth: np.ndarray,
    test_scores: np.ndarray,
) -> RunMetrics:
    """Pick the threshold on validation, then score the test period once with it."""
    threshold = best_f1_threshold(validation_truth, validation_scores)
    return score_period(test_truth, test_scores, threshold)


def score_validation(
    validation_truth: np.ndarray, validation_scores: np.ndarray
) -> RunMetrics:
    """Validation metrics at the threshold chosen on validation itself.

    The F1 here is the best validation F1 (the threshold is tuned on these very
    rows), so it is an upper bound used to compare runs, never a test figure.
    """
    threshold = best_f1_threshold(validation_truth, validation_scores)
    return score_period(validation_truth, validation_scores, threshold)
