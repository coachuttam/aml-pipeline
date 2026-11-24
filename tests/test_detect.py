"""Tests for src/detect.py (top alerts, SHAP reasons, parquet)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import load_dataset
from src.detect import (
    REASON_COLUMNS,
    alert_id_for,
    build_alerts,
    reason_text,
    reasons_for,
    top_positions,
    write_alerts,
)
from src.features import FEATURE_NAMES, build_feature_matrix
from src.model import train_detector


@pytest.fixture
def scored(tiny_csv: Path, monkeypatch):
    monkeypatch.setattr(config, "MAX_TREES", 20)
    dataset = load_dataset(tiny_csv)
    features = build_feature_matrix(dataset.transactions)
    detector = train_detector(features, dataset.truth, dataset.split, FEATURE_NAMES)
    scores = detector.model.predict_proba(features[dataset.split.test])[:, 1]
    alerts = build_alerts(
        dataset.transactions, features, FEATURE_NAMES, scores, dataset.split.test,
        detector, count=30,
    )  # fmt: skip
    return dataset, scores, alerts


def test_top_positions_are_best_first_and_ties_keep_file_order() -> None:
    scores = np.array([0.2, 0.9, 0.9, 0.1])
    assert top_positions(scores, 3).tolist() == [1, 2, 0]


def test_alert_ids_follow_the_rank() -> None:
    assert alert_id_for(1) == "A0001" and alert_id_for(200) == "A0200"


def test_alerts_are_the_highest_scored_test_rows_in_rank_order(scored) -> None:
    dataset, scores, alerts = scored
    assert len(alerts) == 30
    assert alerts["rank"].tolist() == list(range(1, 31))
    assert alerts["score"].is_monotonic_decreasing
    assert alerts["row"].min() >= dataset.split.test.start
    assert alerts["score"].iloc[0] == pytest.approx(scores.max())


def test_every_alert_has_five_reason_codes_with_signed_shap_values(scored) -> None:
    _, _, alerts = scored
    assert REASON_COLUMNS == ["reason_%d" % i for i in range(1, 6)]
    first = alerts.iloc[0]
    for column in REASON_COLUMNS:
        assert " = " in first[column] and first[column].endswith(")")
    assert first["reasons_json"].count('"feature"') == 5


def test_the_alert_table_carries_no_label(scored) -> None:
    _, _, alerts = scored
    assert not [name for name in alerts.columns if "laundering" in name]


def test_reasons_name_the_strongest_features_first() -> None:
    features = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["a", "b", "c"])
    reasons = reasons_for(np.array([[0.1, -0.9, 0.5]]), features, 2)
    assert [r["feature"] for r in reasons[0]] == ["b", "c"]
    assert reason_text(reasons[0][0]) == "b = 2 (-0.90)"


def test_alerts_round_trip_through_parquet(scored, tmp_path: Path) -> None:
    _, _, alerts = scored
    path = write_alerts(alerts, tmp_path)
    assert path.name == "alerts.parquet"
    back = pd.read_parquet(path)
    assert back["alert_id"].tolist() == alerts["alert_id"].tolist()
    assert list(back.columns) == list(alerts.columns)
