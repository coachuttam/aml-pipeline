"""Evidence: the transactions linked to an alert, from the label-free table.

For an alert (sender S pays receiver R at time T) the evidence is:
- hop 1: other transactions that touch S or R within +/- 7 days of T;
- hop 2: transactions that touch the accounts those hop-1 transactions met
  (other than S and R) within the same window.
At most 50 linked transactions are kept, hop 1 first and the nearest in time
first inside each hop. The file records how many existed and whether it was cut.
Strategy "capped" (evidence v2) keeps at most 35 hop-1 and 15 hop-2 rows, so a hub
whose hop 1 alone fills 50 still shows where the money went next.

The input has no label column, so evidence cannot carry one (AGENTS.md rule 2).

Example:
    index = EvidenceIndex(transactions)
    evidence = build_evidence(index, row=5996025, alert_id="A0001")
    evidence["linked_transactions"][0]["hop"]   # 1
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src import config
from src.data import SECONDS_PER_DAY, account_names
from src.evidence_flow import flow_selection
from src.features_base import add_account_codes


class EvidenceIndex:
    """Integer account codes and times for fast lookups, plus the readable table."""

    def __init__(self, transactions: pd.DataFrame) -> None:
        coded = add_account_codes(transactions.copy(deep=False))
        self.transactions = transactions
        self.from_codes = coded["from_code"].to_numpy()
        self.to_codes = coded["to_code"].to_numpy()
        self.seconds = transactions["seconds"].to_numpy()


def touching(index: EvidenceIndex, accounts: np.ndarray, window: slice) -> np.ndarray:
    """Rows inside the window whose sender or receiver is one of `accounts`."""
    senders = np.isin(index.from_codes[window], accounts)
    receivers = np.isin(index.to_codes[window], accounts)
    return np.flatnonzero(senders | receivers) + window.start


def window_rows(index: EvidenceIndex, row: int) -> slice:
    """Rows within +/- the evidence window around transaction `row` (time sorted)."""
    centre = index.seconds[row]
    span = config.EVIDENCE_WINDOW_DAYS * SECONDS_PER_DAY
    low = np.searchsorted(index.seconds, centre - span, side="left")
    high = np.searchsorted(index.seconds, centre + span, side="right")
    return slice(int(low), int(high))


def nearest_first(index: EvidenceIndex, rows: np.ndarray, row: int) -> np.ndarray:
    """`rows` ordered by distance in time from `row`, ties by file order."""
    distance = np.abs(index.seconds[rows] - index.seconds[row])
    return rows[np.lexsort((rows, distance))]


def hop_rows(index: EvidenceIndex, row: int) -> Dict[int, np.ndarray]:
    """Linked rows by hop (nearest first), before the cap. The alert row is excluded."""
    window = window_rows(index, row)
    own = np.array([index.from_codes[row], index.to_codes[row]])
    first = touching(index, own, window)
    first = nearest_first(index, first[first != row], row)
    met = np.unique(np.concatenate([index.from_codes[first], index.to_codes[first]]))
    met = met[~np.isin(met, own)]
    second = touching(index, met, window) if config.EVIDENCE_HOPS >= 2 else first[:0]
    second = second[(second != row) & ~np.isin(second, first)]
    return {1: first, 2: nearest_first(index, second, row)}


def describe(index: EvidenceIndex, rows: np.ndarray, hop: int) -> List[Dict[str, Any]]:
    """The readable fields of each linked transaction."""
    frame = index.transactions.iloc[rows]
    sender, receiver = account_names(frame)
    return [
        {
            "transaction_id": int(frame["edge_id"].iloc[i]),
            "hop": hop,
            "time": frame["timestamp"].iloc[i].strftime("%Y-%m-%d %H:%M"),
            "from_account": sender.iloc[i],
            "to_account": receiver.iloc[i],
            "amount_paid": float(frame["amount_paid"].iloc[i]),
            "payment_currency": str(frame["payment_currency"].iloc[i]),
            "amount_received": float(frame["amount_received"].iloc[i]),
            "receiving_currency": str(frame["receiving_currency"].iloc[i]),
            "payment_format": str(frame["payment_format"].iloc[i]),
        }
        for i in range(len(frame))
    ]


def nearest_document(
    index: EvidenceIndex,
    row: int,
    hop1_cap: Optional[int] = None,
    hop2_cap: Optional[int] = None,
) -> Dict[str, Any]:
    """Strategy "nearest": hop 1 first, then hop 2, the 50 nearest in time.

    With caps (strategy "capped") each hop keeps at most its own cap.
    """
    hops = hop_rows(index, row)
    room = config.EVIDENCE_MAX_EDGES
    kept_first = hops[1][: room if hop1_cap is None else hop1_cap]
    second_room = room - len(kept_first) if hop2_cap is None else hop2_cap
    kept_second = hops[2][:second_room]
    return {
        "linked_transactions": describe(index, kept_first, 1)
        + describe(index, kept_second, 2),
        "counts": {
            "hop1_found": int(len(hops[1])),
            "hop2_found": int(len(hops[2])),
            "included": int(len(kept_first) + len(kept_second)),
            "truncated": bool(
                len(hops[1]) + len(hops[2]) > len(kept_first) + len(kept_second)
            ),
        },
    }


def flow_document(index: EvidenceIndex, row: int) -> Dict[str, Any]:
    """Strategy "flow": 15 nearest hop-1 rows, then 35 along the money trail."""
    nearest = hop_rows(index, row)[1]
    first = nearest[: config.EVIDENCE_HOP1_EDGES]
    skip = np.concatenate([first, [row]])
    chosen, found = flow_selection(index, row, skip, config.EVIDENCE_FLOW_EDGES)
    linked = describe(index, first, 1)
    for direction, (step, _, flow_row) in chosen:
        item = describe(index, np.array([flow_row]), 2)[0]
        item.update({"flow": direction, "flow_step": step})
        linked.append(item)
    return {
        "linked_transactions": linked,
        "counts": {
            "hop1_found": int(len(nearest)),
            **found,
            "included": len(linked),
            "truncated": bool(
                len(nearest) > len(first)
                or found["onward_found"] + found["backward_found"] > len(chosen)
            ),
        },
    }


def build_evidence(
    index: EvidenceIndex, row: int, alert_id: str, strategy: Optional[str] = None
) -> Dict[str, Any]:
    """The evidence document for one alert (JSON-ready)."""
    strategy = strategy or config.EVIDENCE_STRATEGY
    if strategy == config.EVIDENCE_V2_STRATEGY:
        body = nearest_document(
            index, row, config.EVIDENCE_V2_HOP1_EDGES, config.EVIDENCE_V2_HOP2_EDGES
        )
    elif strategy == "nearest":
        body = nearest_document(index, row)
    elif strategy == "flow":
        body = flow_document(index, row)
    else:
        raise ValueError("Unknown evidence strategy %r" % strategy)
    return {
        "alert_id": alert_id,
        "strategy": strategy,
        "alert_transaction": describe(index, np.array([row]), 0)[0],
        "window_days": config.EVIDENCE_WINDOW_DAYS,
        **body,
    }


def write_evidence(document: Dict[str, Any], folder: Path) -> Path:
    """Save one evidence document as `<alert_id>.json`."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (document["alert_id"] + ".json")
    path.write_text(json.dumps(document, indent=2) + "\n")
    return path
