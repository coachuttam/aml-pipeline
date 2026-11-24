"""Tests for src/model.py and src/metrics.py (made-up arrays, small trees)."""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.data import DaySplit
from src.features import FEATURE_NAMES
from src.metrics import best_f1_threshold, score_period
from src.model import resolve_weight, train_detector


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(config, "MAX_TREES", 20)
    rows = 3000
    generator = np.random.RandomState(0)
    features = generator.normal(size=(rows, len(FEATURE_NAMES))).astype(np.float32)
    truth = (features[:, 0] + generator.normal(size=rows) > 2.0).astype(np.int8)
    return features, truth, DaySplit(1800, 2400, rows)


def test_balanced_weight_is_negatives_over_positives() -> None:
    truth = np.array([0] * 90 + [1] * 10)
    assert resolve_weight("balanced", truth) == pytest.approx(9.0)
    assert resolve_weight(0.5, truth) == 0.5


def test_the_detector_picks_the_weight_with_the_best_validation_pr_auc(small) -> None:
    features, truth, split = small
    detector = train_detector(features, truth, split, FEATURE_NAMES)
    tried = detector.validation_pr_auc_by_weight
    assert set(tried) == {str(w) for w in config.CANDIDATE_WEIGHTS}
    winner = max(tried, key=tried.get)
    assert detector.class_weight == pytest.approx(float(winner))


def test_the_threshold_is_the_validation_best_f1_cut_off(small) -> None:
    features, truth, split = small
    detector = train_detector(features, truth, split, FEATURE_NAMES)
    scores = detector.model.predict_proba(features[split.validation])[:, 1]
    expected = best_f1_threshold(truth[split.validation], scores)
    assert detector.threshold == pytest.approx(expected)


def test_test_rows_cannot_change_the_detector(small) -> None:
    features, truth, split = small
    poisoned = features.copy()
    poisoned[split.test] = np.nan
    flipped = truth.copy()
    flipped[split.test] = 1 - flipped[split.test]
    first = train_detector(features, truth, split, FEATURE_NAMES)
    second = train_detector(poisoned, flipped, split, FEATURE_NAMES)
    assert first.threshold == second.threshold
    assert first.class_weight == second.class_weight


def test_period_scores_follow_the_project_two_definitions() -> None:
    truth = np.array([1, 1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.6, 0.8, 0.2, 0.1, 0.05])
    metrics = score_period(truth, scores, 0.5)
    assert metrics.alerts == 3
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == 1.0
    assert metrics.minority_f1 == pytest.approx(0.8)
