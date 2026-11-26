"""Choose which HI-Small alerts go into the fine-tuning set, and split them.

Laundering alerts are taken round-robin across pattern attempts (the best-scored of
each attempt first, at most a few per attempt) so many different patterns appear.
False alarms are the best-scored non-laundering transactions, a few per sender.
Train and validation never share a pattern attempt: whole attempts are assigned,
typology by typology, so every typology reaches both sides when it has 2+ attempts.

This module reads labels on purpose: it builds training data, not the student's
input. Labels never reach the alert table the student sees.

Example:
    chosen = select_alerts(table)            # table = alerts + ground-truth columns
    chosen = assign_splits(chosen)           # adds a "split" column
"""

from __future__ import annotations

import json
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src import config

TRAIN, VALIDATION = "train", "validation"


def laundering_candidates(table: pd.DataFrame) -> pd.DataFrame:
    """Laundering alerts with a known pattern, best round-robin order first."""
    known = table[(table["is_laundering"] == 1) & table["pattern_type"].notna()].copy()
    known = known.sort_values("score", ascending=False, kind="stable")
    known["turn"] = known.groupby("attempt_id").cumcount()
    known = known[known["turn"] < config.FINETUNE_ALERTS_PER_PATTERN]
    return known.sort_values(["turn", "score"], ascending=[True, False], kind="stable")


def false_alarm_candidates(table: pd.DataFrame) -> pd.DataFrame:
    """Best-scored non-laundering alerts, at most a few per sender."""
    clean = table[table["is_laundering"] == 0].sort_values(
        "score", ascending=False, kind="stable"
    )
    turn = clean.groupby("from_account").cumcount()
    return clean[turn < config.FINETUNE_FALSE_ALARMS_PER_SENDER]


def top_feature(reasons_json: str) -> str:
    """The feature of an alert's strongest reason."""
    return json.loads(reasons_json)[0]["feature"]


def take_by_weight(ordered: pd.DataFrame, count: int, weights: Dict[str, float]):
    """`count` rows in preference order, with the top-feature mix of `weights`.

    Each feature gets its share (as far as rows exist); what is missing is filled
    from the best rows left, so the count is met whenever enough rows exist.
    """
    if not weights:
        return ordered.head(count)
    picked = [
        ordered[ordered["top_feature"] == feature].head(int(round(count * share)))
        for feature, share in weights.items()
    ]
    chosen = pd.concat(picked)
    rest = ordered[~ordered.index.isin(chosen.index)]
    return pd.concat([chosen, rest.head(max(count - len(chosen), 0))]).head(count)


def select_alerts(
    table: pd.DataFrame, weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """About FINETUNE_ALERTS alerts, FINETUNE_LAUNDERING_SHARE of them laundering.

    With `weights` (top-feature shares taken from the dev alerts) each class gets the
    same top-reason mix, so the first reason says nothing about the class.
    """
    wanted = config.FINETUNE_ALERTS
    laundering_count = int(round(wanted * config.FINETUNE_LAUNDERING_SHARE))
    table = table.assign(top_feature=table["reasons_json"].map(top_feature))
    laundering = take_by_weight(
        laundering_candidates(table), laundering_count, weights or {}
    )
    false_alarms = take_by_weight(
        false_alarm_candidates(table), wanted - len(laundering), weights or {}
    )
    chosen = pd.concat([laundering, false_alarms]).drop(columns=["turn", "top_feature"])
    return chosen.sort_values("score", ascending=False, kind="stable").reset_index(
        drop=True
    )


def validation_attempts(chosen: pd.DataFrame) -> set:
    """Pattern attempts held out for validation, picked within each typology."""
    generator = np.random.RandomState(config.FINETUNE_SPLIT_SEED)
    held_out = set()
    laundering = chosen[chosen["is_laundering"] == 1]
    for _, group in laundering.groupby("pattern_type"):
        attempts = np.sort(group["attempt_id"].unique())
        generator.shuffle(attempts)
        count = int(round(len(attempts) * config.FINETUNE_VALIDATION_SHARE))
        if len(attempts) >= 2:
            count = max(count, 1)
        held_out.update(int(a) for a in attempts[:count])
    return held_out


def assign_splits(chosen: pd.DataFrame) -> pd.DataFrame:
    """Add `split`: whole pattern attempts to validation; false alarms at random."""
    held_out = validation_attempts(chosen)
    generator = np.random.RandomState(config.FINETUNE_SPLIT_SEED + 1)
    draws = generator.rand(len(chosen))
    is_laundering = (chosen["is_laundering"] == 1).to_numpy()
    in_validation = np.where(
        is_laundering,
        chosen["attempt_id"].isin(held_out).to_numpy(),
        draws < config.FINETUNE_VALIDATION_SHARE,
    )
    result = chosen.copy()
    result["split"] = np.where(in_validation, VALIDATION, TRAIN)
    return result


def pick_sample(chosen: pd.DataFrame, count: int) -> pd.DataFrame:
    """A small batch with the same laundering share, typologies spread round-robin."""
    if count >= len(chosen):
        return chosen
    laundering_count = int(round(count * config.FINETUNE_LAUNDERING_SHARE))
    generator = np.random.RandomState(config.FINETUNE_SAMPLE_SEED)
    laundering = chosen[chosen["is_laundering"] == 1]
    groups = [g.sample(frac=1, random_state=generator) for _, g in
              laundering.groupby("pattern_type")]  # fmt: skip
    picked = []
    for turn in range(max(len(g) for g in groups)):
        picked += [g.iloc[turn] for g in groups if turn < len(g)]
    first = pd.DataFrame(picked[:laundering_count])
    others = chosen[chosen["is_laundering"] == 0]
    rest = others.sample(n=count - len(first), random_state=generator)
    return pd.concat([first, rest]).sort_values("score", ascending=False)
