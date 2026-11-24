"""Train the LightGBM detector and pick its alert threshold on validation.

The settings are the Phase E final ones from aml-gnn-transfer: all training labels
(label-mask seed 0), up to 2000 trees with early stopping on validation PR-AUC,
and the class weight that gives the best validation PR-AUC. The alert threshold
is the best-F1 cut-off on the validation period. The test period is only scored,
never used to choose anything (AGENTS.md rule 1).

Example:
    detector = train_detector(features, truth, split)
    scores = detector.model.predict_proba(features[split.test])[:, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.metrics import average_precision_score

from src import config
from src.data import DaySplit
from src.metrics import best_f1_threshold


@dataclass(frozen=True)
class Detector:
    """A trained model, the weight that won, and the validation threshold."""

    model: Any
    class_weight: float
    threshold: float
    validation_pr_auc_by_weight: Dict[str, float]


def resolve_weight(candidate: object, truth: np.ndarray) -> float:
    """ "balanced" means negatives / positives among the training rows."""
    if candidate != "balanced":
        return float(candidate)  # type: ignore[arg-type]
    positives = max(int(truth.sum()), 1)
    return (len(truth) - positives) / positives


def fit_one(
    features: np.ndarray,
    truth: np.ndarray,
    split: DaySplit,
    weight: float,
    names: list,
) -> Any:
    """LightGBM on the training period; early stopping watches validation PR-AUC."""
    import lightgbm

    model = lightgbm.LGBMClassifier(
        n_estimators=config.MAX_TREES,
        learning_rate=config.LEARNING_RATE,
        num_leaves=config.LEAVES,
        scale_pos_weight=weight,
        random_state=config.MODEL_SEED + config.RUN_SEED,
        verbose=-1,
    )
    validation = split.validation
    model.fit(
        features[split.train],
        truth[split.train],
        eval_set=[(features[validation], truth[validation])],
        eval_metric="average_precision",
        feature_name=names,
        callbacks=[
            lightgbm.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False)
        ],
    )
    return model


def train_detector(
    features: np.ndarray, truth: np.ndarray, split: DaySplit, names: list
) -> Detector:
    """Try each class weight, keep the best on validation PR-AUC, set the threshold."""
    validation = split.validation
    tried: Dict[str, float] = {}
    best = (-1.0, None, 1.0)
    for candidate in config.CANDIDATE_WEIGHTS:
        weight = resolve_weight(candidate, truth[split.train])
        model = fit_one(features, truth, split, weight, names)
        scores = model.predict_proba(features[validation])[:, 1]
        pr_auc = float(average_precision_score(truth[validation], scores))
        tried[str(candidate)] = pr_auc
        if pr_auc > best[0]:
            best = (pr_auc, model, weight)
    _, model, weight = best
    validation_scores = model.predict_proba(features[validation])[:, 1]
    threshold = best_f1_threshold(truth[validation], validation_scores)
    return Detector(model, weight, threshold, tried)
