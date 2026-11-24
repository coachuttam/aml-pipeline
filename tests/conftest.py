"""Shared test data: a small made-up IBM-style CSV (no real data, no Kaggle)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

RAW_HEADER = (
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,"
    "Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
)
DAYS = 10
ROWS_PER_DAY = 300


def raw_line(
    minute: int, sender: str, receiver: str, amount: float = 100.0, label: int = 0
) -> str:
    """One CSV line; `minute` counts from 2022/09/01 00:00."""
    day, minute_of_day = divmod(minute, 24 * 60)
    stamp = "2022/09/%02d %02d:%02d" % (
        1 + day, minute_of_day // 60, minute_of_day % 60
    )  # fmt: skip
    return "%s,10,%s,10,%s,%.2f,US Dollar,%.2f,US Dollar,Cheque,%d\n" % (
        stamp, sender, receiver, amount, amount, label,
    )  # fmt: skip


def write_raw(path: Path, lines) -> Path:
    path.write_text(RAW_HEADER + "".join(lines))
    return path


def made_up_lines(seed: int = 0):
    """Ten days of payments among 60 accounts; large payments are often laundering."""
    generator = np.random.RandomState(seed)
    lines = []
    for day in range(DAYS):
        for _ in range(ROWS_PER_DAY):
            minute = day * 24 * 60 + int(generator.randint(0, 24 * 60))
            sender, receiver = generator.randint(0, 60, size=2)
            amount = float(np.round(generator.lognormal(5, 1.5), 2))
            label = int(amount > 1500 and generator.rand() < 0.5)
            lines.append((minute, "A%02d" % sender, "A%02d" % receiver, amount, label))
    lines.sort(key=lambda line: line[0])
    return [raw_line(*line) for line in lines]


@pytest.fixture
def tiny_csv(tmp_path: Path) -> Path:
    return write_raw(tmp_path / "LI-Small_Trans.csv", made_up_lines())
