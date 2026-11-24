"""Ground truth for evaluation only: is the alert laundering, and which pattern?

This module is the only place labels meet alerts. Its output
(`artifacts/ground_truth.parquet`) is kept apart from `alerts.parquet` and the
evidence files, and the LLM step never reads it (AGENTS.md rule 2).

`LI-Small_Patterns.txt` (from IBM's Kaggle dataset) lists every laundering
transaction under a header such as "BEGIN LAUNDERING ATTEMPT - FAN-OUT". When the
file is missing, pattern types are left empty and the summary says so.

Example:
    patterns = read_patterns(config.PATTERNS_FILE)        # {} if the file is missing
    table = ground_truth_table(alerts, transactions, truth, patterns)
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd

from src import config

BEGIN_PATTERN = re.compile(r"BEGIN LAUNDERING ATTEMPT - ([A-Z\-]+)")
AMOUNT_DECIMALS = 2
TIME_FORMAT = "%Y/%m/%d %H:%M"
PatternKey = Tuple[str, str, str, str, str, float, float, str, str]
# raw columns: time, from bank, from account, to bank, to account, amount received,
# receiving currency, amount paid, payment currency, payment format, is laundering
TIME, FROM_BANK, FROM_ACCOUNT, TO_BANK, TO_ACCOUNT = range(5)
AMOUNT_RECEIVED, RECEIVING_CURRENCY, AMOUNT_PAID, PAYMENT_CURRENCY, FORMAT = range(
    5, 10
)


def make_key(
    time: str, from_bank: str, from_account: str, to_bank: str, to_account: str,
    amount_received: float, amount_paid: float, currency: str, payment_format: str,
) -> PatternKey:  # fmt: skip
    """What identifies a transaction in both files (there is no id column)."""
    return (
        time, from_bank, from_account, to_bank, to_account,
        round(float(amount_received), AMOUNT_DECIMALS),
        round(float(amount_paid), AMOUNT_DECIMALS), currency, payment_format,
    )  # fmt: skip


def read_patterns(path: Path) -> Dict[PatternKey, Tuple[str, int]]:
    """Map each listed laundering transaction to (pattern type, attempt number)."""
    if not path.exists():
        return {}
    found: Dict[PatternKey, Tuple[str, int]] = {}
    pattern, attempt = "", -1
    for line in path.read_text().splitlines():
        header = BEGIN_PATTERN.search(line)
        if header:
            pattern, attempt = header.group(1).replace("-", "_"), attempt + 1
        elif attempt >= 0 and line.strip() and not line.startswith("END"):
            cells = line.split(",")
            if len(cells) <= FORMAT:  # not a transaction line
                continue
            key = make_key(
                cells[TIME], cells[FROM_BANK], cells[FROM_ACCOUNT], cells[TO_BANK],
                cells[TO_ACCOUNT], float(cells[AMOUNT_RECEIVED]),
                float(cells[AMOUNT_PAID]), cells[PAYMENT_CURRENCY], cells[FORMAT],
            )  # fmt: skip
            found.setdefault(key, (pattern, attempt))
    return found


def key_of_row(transactions: pd.DataFrame, row: int) -> PatternKey:
    """The pattern-file key of one transaction row."""
    line = transactions.iloc[row]
    return make_key(
        line["timestamp"].strftime(TIME_FORMAT), str(line["from_bank"]),
        str(line["from_account"]), str(line["to_bank"]), str(line["to_account"]),
        line["amount_received"], line["amount_paid"],
        str(line["payment_currency"]), str(line["payment_format"]),
    )  # fmt: skip


def ground_truth_table(
    alerts: pd.DataFrame,
    transactions: pd.DataFrame,
    truth: np.ndarray,
    patterns: Dict[PatternKey, Tuple[str, int]],
) -> pd.DataFrame:
    """alert_id, is_laundering and (if the patterns file exists) pattern type."""
    rows = alerts["row"].to_numpy()
    types, attempts = [], []
    for row in rows:
        kind, attempt = patterns.get(key_of_row(transactions, int(row)), (None, None))
        types.append(kind)
        attempts.append(attempt)
    return pd.DataFrame(
        {
            "alert_id": alerts["alert_id"].to_numpy(),
            "edge_id": alerts["edge_id"].to_numpy(),
            "is_laundering": truth[rows].astype(int),
            "pattern_type": types,
            "attempt_id": attempts,
        }
    )


def write_ground_truth(table: pd.DataFrame, folder: Path) -> Path:
    """Save the ground-truth table as parquet in `folder`."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / config.GROUND_TRUTH_FILE
    table.to_parquet(path, index=False)
    return path


def laundering_share(table: pd.DataFrame) -> float:
    """Share of the alerts that are laundering (precision at the number of alerts)."""
    return float(table["is_laundering"].mean()) if len(table) else 0.0


def pattern_members(
    patterns: Dict[PatternKey, Tuple[str, int]], transactions: pd.DataFrame
) -> Dict[int, Set[int]]:
    """For each pattern attempt, the edge ids of all its transactions in the data.

    Only rows sharing a minute with a listed transaction are compared, so the
    6.9 million rows are not all keyed.
    """
    if not patterns:
        return {}
    minutes = pd.to_datetime(sorted({key[0] for key in patterns}), format=TIME_FORMAT)
    candidates = transactions[transactions["timestamp"].isin(minutes)]
    members: Dict[int, Set[int]] = defaultdict(set)
    columns = [
        candidates[name]
        for name in (
            "edge_id", "timestamp", "from_bank", "from_account", "to_bank",
            "to_account", "amount_received", "amount_paid", "payment_currency",
            "payment_format",
        )
    ]  # fmt: skip
    for edge, stamp, fb, fa, tb, ta, received, paid, currency, kind in zip(*columns):
        key = make_key(
            stamp.strftime(TIME_FORMAT), str(fb), str(fa), str(tb), str(ta),
            received, paid, str(currency), str(kind),
        )  # fmt: skip
        if key in patterns:
            members[patterns[key][1]].add(int(edge))
    return dict(members)
