"""Per-transaction features for the detector (copied from aml-gnn-transfer).

Copied from Project 2 (aml-ibm-experiment/src/ibm/features.py), which was not
changed; `add_account_codes` comes from its loader.py, rewritten to use less
memory. No feature reads a label: the columns come from timestamps, accounts,
amounts and formats only, so the label masks of the label-scarcity runs cannot
leak into them. Each one is computed from the past only.

For each transaction we describe the sender's (and receiver's) behaviour up to
that moment, never after it, so the model cannot peek at the future.
Amounts are in each row's own currency; the data has no exchange rates.

Example:
    table = build_features(transactions)   # needs account codes (see loader)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

SECONDS_PER_HOUR = 3600
VELOCITY_WINDOWS: Dict[str, int] = {
    "1h": SECONDS_PER_HOUR,
    "24h": 24 * SECONDS_PER_HOUR,
    "7d": 7 * 24 * SECONDS_PER_HOUR,
}
# Larger than any timestamp plus window, so one account's block never overlaps the next.
ACCOUNT_BLOCK_SIZE = 10**10
ROUND_AMOUNT_STEP = 100  # multiples of 100 look "made up" more often than real prices
VERY_ROUND_AMOUNT_STEP = 1_000

FEATURE_COLUMNS: List[str] = (
    ["sender_sent_%s" % name for name in VELOCITY_WINDOWS]
    + ["receiver_received_%s" % name for name in VELOCITY_WINDOWS]
    + [
        "sender_out_degree",
        "receiver_in_degree",
        "sender_in_out_ratio",
        "is_round_amount",
        "is_very_round_amount",
        "is_cross_bank",
        "is_currency_change",
        "log_amount_paid",
        "payment_format_code",
        "payment_currency_code",
    ]
)


def shared_codes(first: pd.Series, second: pd.Series) -> tuple:
    """Integer codes for two category columns, numbered against one shared list."""
    both = first.astype(str).astype("category").cat.categories
    both = both.union(second.astype(str).astype("category").cat.categories)
    first_codes = first.astype(str).astype(pd.CategoricalDtype(both)).cat.codes
    second_codes = second.astype(str).astype(pd.CategoricalDtype(both)).cat.codes
    return first_codes.to_numpy().astype(np.int64), second_codes.to_numpy(), len(both)


def add_account_codes(frame: pd.DataFrame) -> pd.DataFrame:
    """Add `from_code` and `to_code`: one number per (bank, account).

    Same result as Project 2's version, which joined bank and account as text; this
    one joins their integer codes, which needs far less memory on 7 million rows.
    """
    bank_from, bank_to, bank_count = shared_codes(frame["from_bank"], frame["to_bank"])
    account_from, account_to, account_count = shared_codes(
        frame["from_account"], frame["to_account"]
    )
    sender_keys = bank_from * account_count + account_from
    receiver_keys = bank_to.astype(np.int64) * account_count + account_to
    codes, _ = pd.factorize(np.concatenate([sender_keys, receiver_keys]))
    frame["from_code"] = codes[: len(frame)]
    frame["to_code"] = codes[len(frame) :]
    return frame


def _seconds(frame: pd.DataFrame) -> np.ndarray:
    return (
        (frame["timestamp"] - frame["timestamp"].min())
        .dt.total_seconds()
        .to_numpy()
        .astype(np.int64)
    )


def cumulative_up_to(
    stream_codes: np.ndarray,
    stream_seconds: np.ndarray,
    stream_values: np.ndarray,
    query_codes: np.ndarray,
    query_seconds: np.ndarray,
) -> np.ndarray:
    """For each query, the `stream_values` total of the same account up to that second.

    Example: two payments by account 7 (values 1 and 1, at seconds 10 and 50)
    answers a query (account 7, second 60) with 2.0.
    """
    stream_keys = stream_codes.astype(np.int64) * ACCOUNT_BLOCK_SIZE + stream_seconds
    order = np.argsort(stream_keys, kind="stable")
    sorted_keys = stream_keys[order]
    running_total = np.concatenate(
        [[0.0], np.cumsum(stream_values[order].astype(float))]
    )
    query_keys = query_codes.astype(np.int64) * ACCOUNT_BLOCK_SIZE + query_seconds
    position = np.searchsorted(sorted_keys, query_keys, side="right")
    block_start = np.searchsorted(
        sorted_keys, query_codes.astype(np.int64) * ACCOUNT_BLOCK_SIZE, side="left"
    )
    return running_total[position] - running_total[block_start]


def add_velocity(frame: pd.DataFrame, table: pd.DataFrame) -> None:
    """Payments sent and received in the last 1 hour, 24 hours and 7 days."""
    seconds = _seconds(frame)
    ones = np.ones(len(frame))
    from_codes, to_codes = frame["from_code"].to_numpy(), frame["to_code"].to_numpy()
    for name, window in VELOCITY_WINDOWS.items():
        now_sent = cumulative_up_to(from_codes, seconds, ones, from_codes, seconds)
        before_sent = cumulative_up_to(
            from_codes, seconds, ones, from_codes, seconds - window - 1
        )
        table["sender_sent_%s" % name] = now_sent - before_sent
        now_received = cumulative_up_to(to_codes, seconds, ones, to_codes, seconds)
        before_received = cumulative_up_to(
            to_codes, seconds, ones, to_codes, seconds - window - 1
        )
        table["receiver_received_%s" % name] = now_received - before_received


def _running_distinct_partners(
    own_codes: pd.Series, partner_codes: pd.Series
) -> np.ndarray:
    """For each row, how many different partners that account has dealt with so far."""
    first_time_pair = ~pd.DataFrame(
        {"own": own_codes, "partner": partner_codes}
    ).duplicated()
    return first_time_pair.groupby(own_codes).cumsum().to_numpy()


def add_degree(frame: pd.DataFrame, table: pd.DataFrame) -> None:
    """Distinct accounts the sender paid so far, and that paid the receiver."""
    table["sender_out_degree"] = _running_distinct_partners(
        frame["from_code"], frame["to_code"]
    )
    table["receiver_in_degree"] = _running_distinct_partners(
        frame["to_code"], frame["from_code"]
    )


def add_in_out_ratio(frame: pd.DataFrame, table: pd.DataFrame) -> None:
    """Money sent so far over money received so far (high = pass-through)."""
    seconds = _seconds(frame)
    amounts = frame["amount_paid"].to_numpy()
    from_codes = frame["from_code"].to_numpy()
    sent = cumulative_up_to(from_codes, seconds, amounts, from_codes, seconds)
    received = cumulative_up_to(
        frame["to_code"].to_numpy(), seconds, amounts, from_codes, seconds
    )
    table["sender_in_out_ratio"] = sent / (received + 1.0)


def add_simple_flags(frame: pd.DataFrame, table: pd.DataFrame) -> None:
    """Round amounts, cross-bank, currency change, and the amount and format columns."""
    amount = frame["amount_paid"]
    table["is_round_amount"] = (
        (amount % ROUND_AMOUNT_STEP == 0) & (amount > 0)
    ).astype(int)
    table["is_very_round_amount"] = (
        (amount % VERY_ROUND_AMOUNT_STEP == 0) & (amount > 0)
    ).astype(int)
    table["is_cross_bank"] = (
        frame["from_bank"].astype(str) != frame["to_bank"].astype(str)
    ).astype(int)
    table["is_currency_change"] = (
        frame["receiving_currency"].astype(str) != frame["payment_currency"].astype(str)
    ).astype(int)
    table["log_amount_paid"] = np.log1p(amount)
    table["payment_format_code"] = frame["payment_format"].astype("category").cat.codes
    table["payment_currency_code"] = (
        frame["payment_currency"].astype("category").cat.codes
    )


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """The full feature table, one row per transaction, in the same order as `frame`."""
    table = pd.DataFrame(index=frame.index)
    add_velocity(frame, table)
    add_degree(frame, table)
    add_in_out_ratio(frame, table)
    add_simple_flags(frame, table)
    return table[FEATURE_COLUMNS]
