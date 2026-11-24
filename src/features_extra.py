"""Extra graph features (copied from aml-gnn-transfer, src/lgbm_features_extra.py).

Project 2's 16 features were weak on LI-Small, so these add what a bank would
look at next: how many new counterparties an account met in a day or a week
(fan-out and fan-in), how many accounts the receiver pays on (a second hop),
how busy an account has been in total, and how large a payment is next to the
account's usual one. Like the originals, each feature is computed from the past
only and never reads a label.

Example:
    table = build_extra_features(frame)   # frame needs from_code and to_code
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.features_base import VELOCITY_WINDOWS, cumulative_up_to

NEW_PARTNER_WINDOWS = {"24h": VELOCITY_WINDOWS["24h"], "7d": VELOCITY_WINDOWS["7d"]}
EXTRA_COLUMNS: List[str] = (
    ["is_first_time_pair"]
    + ["sender_new_partners_%s" % name for name in NEW_PARTNER_WINDOWS]
    + ["receiver_new_senders_%s" % name for name in NEW_PARTNER_WINDOWS]
    + [
        "receiver_out_degree",
        "sender_in_degree",
        "sender_transaction_count",
        "receiver_transaction_count",
        "sender_mean_amount_sent",
        "receiver_mean_amount_received",
        "amount_vs_sender_mean",
        "amount_vs_receiver_mean",
    ]
)


def elapsed_seconds(frame: pd.DataFrame) -> np.ndarray:
    """Seconds since the first transaction, as whole numbers."""
    start = frame["timestamp"].min()
    return (frame["timestamp"] - start).dt.total_seconds().to_numpy().astype(np.int64)


def first_time_pairs(frame: pd.DataFrame) -> np.ndarray:
    """1.0 on the first payment of each (sender, receiver) pair, else 0.0."""
    pairs = pd.DataFrame({"a": frame["from_code"], "b": frame["to_code"]})
    return (~pairs.duplicated()).to_numpy().astype(float)


def windowed(codes, seconds, values, window: int) -> np.ndarray:
    """Per account, the sum of `values` over the last `window` seconds."""
    now = cumulative_up_to(codes, seconds, values, codes, seconds)
    before = cumulative_up_to(codes, seconds, values, codes, seconds - window - 1)
    return now - before


def add_new_partner_counts(frame, table, seconds, new_pair) -> None:
    """New receivers a sender met, and new senders a receiver met, per window."""
    from_codes, to_codes = frame["from_code"].to_numpy(), frame["to_code"].to_numpy()
    for name, window in NEW_PARTNER_WINDOWS.items():
        table["sender_new_partners_%s" % name] = windowed(
            from_codes, seconds, new_pair, window
        )
        table["receiver_new_senders_%s" % name] = windowed(
            to_codes, seconds, new_pair, window
        )


def add_second_hop(frame, table, seconds, new_pair) -> None:
    """How many accounts the receiver already pays, and how many pay the sender."""
    from_codes, to_codes = frame["from_code"].to_numpy(), frame["to_code"].to_numpy()
    table["receiver_out_degree"] = cumulative_up_to(
        from_codes, seconds, new_pair, to_codes, seconds
    )
    table["sender_in_degree"] = cumulative_up_to(
        to_codes, seconds, new_pair, from_codes, seconds
    )


def add_amount_history(frame, table, seconds) -> None:
    """Transaction counts, mean amounts per account, and this amount against them."""
    from_codes, to_codes = frame["from_code"].to_numpy(), frame["to_code"].to_numpy()
    amounts = frame["amount_paid"].to_numpy().astype(float)
    ones = np.ones(len(frame))
    for role, codes in (("sender", from_codes), ("receiver", to_codes)):
        count = cumulative_up_to(codes, seconds, ones, codes, seconds)
        total = cumulative_up_to(codes, seconds, amounts, codes, seconds)
        mean = total / np.maximum(count, 1.0)
        table["%s_transaction_count" % role] = count
        table[
            "%s_mean_amount_%s" % (role, "sent" if role == "sender" else "received")
        ] = mean
        table["amount_vs_%s_mean" % role] = amounts / (mean + 1.0)


def build_extra_features(frame: pd.DataFrame) -> pd.DataFrame:
    """The extra feature table, one row per transaction, in the order of `frame`."""
    table = pd.DataFrame(index=frame.index)
    seconds = elapsed_seconds(frame)
    new_pair = first_time_pairs(frame)
    table["is_first_time_pair"] = new_pair
    add_new_partner_counts(frame, table, seconds, new_pair)
    add_second_hop(frame, table, seconds, new_pair)
    add_amount_history(frame, table, seconds)
    return table[EXTRA_COLUMNS]
