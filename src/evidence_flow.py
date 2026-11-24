"""Follow the money: the transactions that lead into and out of an alert.

For an alert (S pays R at time T) the evidence is:
- up to 15 hop-1 transactions: other payments touching S or R, nearest in time;
- up to 35 hop-2 flow transactions along the money trail, within 7 days of T:
  - onward: payments R sent after T (step 1), then payments sent by the accounts
    R paid, after they received (step 2);
  - backward: payments S received before T (step 1), then payments received by
    the accounts that paid S, before they paid (step 2).
Onward and backward take turns, step 1 before step 2 and nearest in time first,
so neither side crowds the other out. Rows already among the 15 are skipped.

Like the rest of the evidence code it sees no label.

Example:
    pool = onward_pool(index, row)       # [(step, distance, row), ...] best first
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src import config
from src.data import SECONDS_PER_DAY

FlowRow = Tuple[int, int, int]  # (step along the trail, seconds from alert, row)


def window_bounds(seconds: np.ndarray, row: int) -> Tuple[int, int, int, int]:
    """Row positions: window start, first row at T, first row after T, window end."""
    centre = seconds[row]
    span = config.EVIDENCE_WINDOW_DAYS * SECONDS_PER_DAY
    low = int(np.searchsorted(seconds, centre - span, side="left"))
    before_end = int(np.searchsorted(seconds, centre, side="left"))
    after_start = int(np.searchsorted(seconds, centre, side="right"))
    high = int(np.searchsorted(seconds, centre + span, side="right"))
    return low, before_end, after_start, high


def matching(
    codes: np.ndarray, start: int, stop: int, accounts: np.ndarray
) -> np.ndarray:
    """Rows in [start, stop) whose code is one of `accounts`."""
    return np.flatnonzero(np.isin(codes[start:stop], accounts)) + start


def extreme_per_account(
    accounts: np.ndarray, times: np.ndarray, earliest: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Unique accounts, and each one's earliest (or latest) time."""
    unique, inverse = np.unique(accounts, return_inverse=True)
    best = np.full(len(unique), times.max() + 1 if earliest else times.min() - 1)
    (np.minimum if earliest else np.maximum).at(best, inverse, times)
    return unique, best


def ranked(index, rows: np.ndarray, step: int, row: int) -> List[FlowRow]:
    """(step, distance, row) tuples, nearest in time first."""
    distance = np.abs(index.seconds[rows] - index.seconds[row])
    return [(step, int(d), int(r)) for d, r in sorted(zip(distance, rows))]


def onward_pool(index, row: int) -> List[FlowRow]:
    """Payments R sent after T (step 1), then those of the accounts R paid (step 2)."""
    _, _, after, high = window_bounds(index.seconds, row)
    first = matching(index.from_codes, after, high, index.to_codes[row : row + 1])
    if len(first) == 0:
        return []
    paid, received_at = extreme_per_account(
        index.to_codes[first], index.seconds[first], earliest=True
    )
    candidates = matching(index.from_codes, after, high, paid)
    slot = np.searchsorted(paid, index.from_codes[candidates])
    second = candidates[index.seconds[candidates] > received_at[slot]]
    second = second[~np.isin(second, first)]
    return ranked(index, first, 1, row) + ranked(index, second, 2, row)


def backward_pool(index, row: int) -> List[FlowRow]:
    """Payments S received before T (step 1), then those its payers got (step 2)."""
    low, before_end, _, _ = window_bounds(index.seconds, row)
    first = matching(index.to_codes, low, before_end, index.from_codes[row : row + 1])
    if len(first) == 0:
        return []
    payers, paid_at = extreme_per_account(
        index.from_codes[first], index.seconds[first], earliest=False
    )
    candidates = matching(index.to_codes, low, before_end, payers)
    slot = np.searchsorted(payers, index.to_codes[candidates])
    second = candidates[index.seconds[candidates] < paid_at[slot]]
    second = second[~np.isin(second, first)]
    return ranked(index, first, 1, row) + ranked(index, second, 2, row)


def take_turns(
    onward: List[FlowRow], backward: List[FlowRow], room: int
) -> List[Tuple[str, FlowRow]]:
    """Alternate the two pools (step 1 before step 2) until `room` rows are chosen."""
    pools = {"onward": sorted(onward), "backward": sorted(backward)}
    chosen: List[Tuple[str, FlowRow]] = []
    taken = {"onward": 0, "backward": 0}
    while len(chosen) < room and any(taken[k] < len(p) for k, p in pools.items()):
        for name in ("onward", "backward"):
            if len(chosen) < room and taken[name] < len(pools[name]):
                chosen.append((name, pools[name][taken[name]]))
                taken[name] += 1
    return chosen


def flow_selection(
    index, row: int, skip: np.ndarray, room: int
) -> Tuple[List[Tuple[str, FlowRow]], Dict[str, int]]:
    """The chosen flow rows (not in `skip`) and how many each direction offered."""
    skipped = set(skip.tolist())
    forward = [t for t in onward_pool(index, row) if t[2] not in skipped]
    backward = [t for t in backward_pool(index, row) if t[2] not in skipped]
    found = {"onward_found": len(forward), "backward_found": len(backward)}
    return take_turns(forward, backward, room), found
