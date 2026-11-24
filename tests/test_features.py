"""Tests for src/features.py (feature matrix and its cache)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data import load_dataset
from src.features import FEATURE_NAMES, build_feature_matrix, load_or_build
from tests.conftest import made_up_lines, write_raw


def test_the_matrix_has_one_column_per_feature_name(tiny_csv: Path) -> None:
    dataset = load_dataset(tiny_csv)
    matrix = build_feature_matrix(dataset.transactions)
    assert matrix.shape == (len(dataset.transactions), len(FEATURE_NAMES))
    assert len(FEATURE_NAMES) == 29 and matrix.dtype == np.float32


def test_no_feature_is_named_after_a_label() -> None:
    assert not [name for name in FEATURE_NAMES if "laundering" in name]


def test_features_do_not_change_when_labels_change(tmp_path: Path) -> None:
    lines = made_up_lines()
    flipped = [
        line[:-2] + ("0" if line.endswith("1\n") else "1") + "\n" for line in lines
    ]
    first = load_dataset(write_raw(tmp_path / "one.csv", lines))
    second = load_dataset(write_raw(tmp_path / "two.csv", flipped))
    assert not np.array_equal(first.truth, second.truth)
    assert np.array_equal(
        build_feature_matrix(first.transactions),
        build_feature_matrix(second.transactions),
    )


def test_the_cache_is_reused_while_the_raw_file_is_unchanged(
    tiny_csv: Path, tmp_path: Path, monkeypatch
) -> None:
    dataset = load_dataset(tiny_csv)
    cache = tmp_path / "cache"
    first = load_or_build(dataset.transactions, cache, tiny_csv)
    monkeypatch.setattr(
        "src.features.build_feature_matrix",
        lambda frame: (_ for _ in ()).throw(AssertionError("rebuilt")),
    )
    assert np.array_equal(load_or_build(dataset.transactions, cache, tiny_csv), first)


def test_a_changed_raw_file_makes_the_cache_rebuild(
    tiny_csv: Path, tmp_path: Path
) -> None:
    dataset = load_dataset(tiny_csv)
    cache = tmp_path / "cache"
    load_or_build(dataset.transactions, cache, tiny_csv)
    tiny_csv.write_text(tiny_csv.read_text() + "\n")  # size changes
    rebuilt = load_or_build(dataset.transactions, cache, tiny_csv)
    assert rebuilt.shape[0] == len(dataset.transactions)
