"""The detector's feature matrix: Project 2's graph features plus 13 extra ones.

Both sets are copied from aml-gnn-transfer (`features_base.py`, `features_extra.py`).
They use timestamps, accounts, amounts and formats from the past only, never a
label, and the transactions passed in here have no label column at all. The
matrix (6.9 million rows) takes a few minutes, so it is cached under `cache/`.

Example:
    matrix = load_or_build(dataset.transactions, CACHE_DIR / "features", raw_csv)
    matrix.shape   # (6924049, 29)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.features_base import FEATURE_COLUMNS, add_account_codes, build_features
from src.features_extra import EXTRA_COLUMNS, build_extra_features

FEATURE_NAMES: List[str] = FEATURE_COLUMNS + EXTRA_COLUMNS
MATRIX_FILE = "features.npy"
META_FILE = "features_meta.json"


def build_feature_matrix(transactions: pd.DataFrame) -> np.ndarray:
    """One row per transaction, in `FEATURE_NAMES` order, as float32."""
    start = time.time()
    frame = add_account_codes(transactions.copy(deep=False))  # adds code columns only
    base = build_features(frame).to_numpy().astype(np.float32)
    extra = build_extra_features(frame).to_numpy().astype(np.float32)
    print(
        "Built %d features in %.0f s"
        % (base.shape[1] + extra.shape[1], time.time() - start)
    )
    return np.concatenate([base, extra], axis=1)


def source_stamp(raw_csv: Path) -> dict:
    """What the cached matrix was built from: the raw file's size."""
    return {"raw_bytes": raw_csv.stat().st_size, "features": FEATURE_NAMES}


def load_or_build(
    transactions: pd.DataFrame, cache_dir: Path, raw_csv: Path, rebuild: bool = False
) -> np.ndarray:
    """The cached matrix if it matches the raw file and row count, else a new one."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path, matrix_path = cache_dir / META_FILE, cache_dir / MATRIX_FILE
    stamp = source_stamp(raw_csv)
    if not rebuild and meta_path.exists() and matrix_path.exists():
        if json.loads(meta_path.read_text()) == stamp:
            matrix = np.load(matrix_path)
            if len(matrix) == len(transactions):
                return matrix
    matrix = build_feature_matrix(transactions)
    np.save(matrix_path, matrix)
    meta_path.write_text(json.dumps(stamp))
    return matrix
