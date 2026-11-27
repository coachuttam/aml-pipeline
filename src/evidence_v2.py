"""Evidence v2 ("capped": at most 35 hop-1 + 15 hop-2 rows) against v1, per alert set.

Dev alerts are hubs: v1 kept the 50 nearest rows hop 1 first, so dev evidence was 98%
hop 1 and showed almost no onward money (train 77% / 21%). v2 caps each hop. This
module measures the change only (evaluation, reads ground truth):

- hop mix: share of rows per hop, mean rows per alert, alerts with no hop-2 row;
- pattern coverage: of laundering alerts with a known pattern, the mean share of the
  pattern's transactions in the evidence, and how many show at least
  MIN_VISIBLE_PATTERN_TXNS of them ("visible");
- the rule baseline (src/typology_rules.py) with settings tuned on dev only.

Example:
    hop_mix(documents)   # {"hop_share": {"1": 0.7, "2": 0.3}, ...}
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Set

import pandas as pd

from src import config
from src.evaluate import pattern_coverage
from src.typology_rules import RuleSettings, detect_typology, score_rules


def hop_mix(documents: List[Dict]) -> Dict:
    """Share of linked rows per hop, mean rows per alert, alerts without hop 2."""
    hops: Counter = Counter()
    for document in documents:
        hops.update(t["hop"] for t in document["linked_transactions"])
    total = max(sum(hops.values()), 1)
    no_hop2 = sum(
        not any(t["hop"] == 2 for t in d["linked_transactions"]) for d in documents
    )
    return {
        "alerts": len(documents),
        "hop_share": {str(h): round(n / total, 3) for h, n in sorted(hops.items())},
        "mean_linked": round(total / max(len(documents), 1), 1),
        "alerts_without_hop2": no_hop2,
    }


def coverage_summary(
    documents: List[Dict], truth: pd.DataFrame, members: Dict[int, Set[int]]
) -> Dict:
    """Mean coverage and found rows, and how many alerts show their pattern."""
    rows = pattern_coverage(documents, truth, members)
    visible = rows["found"] >= config.MIN_VISIBLE_PATTERN_TXNS
    return {
        "laundering_with_pattern": len(rows),
        "mean_coverage": round(float(rows["coverage"].mean()), 3),
        "mean_found": round(float(rows["found"].mean()), 2),
        "visible": int(visible.sum()),
        "visible_by_type": rows[visible]["pattern_type"].value_counts().to_dict(),
    }


def visible_flags(
    documents: List[Dict], truth: pd.DataFrame, members: Dict[int, Set[int]]
) -> List[Dict]:
    """Truth rows with pattern_visible from this evidence (for scoring rules)."""
    rows = pattern_coverage(documents, truth, members)
    found = rows.set_index("alert_id")["found"]
    flags = truth["alert_id"].map(found).fillna(0) >= config.MIN_VISIBLE_PATTERN_TXNS
    return truth.assign(pattern_visible=flags.to_numpy()).to_dict("records")


def rule_scores(
    documents: List[Dict], truth: List[Dict], settings: RuleSettings
) -> Dict:
    """Rule baseline and the always-GATHER_SCATTER constant on one evidence set."""
    evidence = {d["alert_id"]: d for d in documents}
    predicted = {i: detect_typology(e, settings) for i, e in evidence.items()}
    constant = {i: "GATHER_SCATTER" for i in evidence}
    return {
        "rules": score_rules(predicted, truth),
        "always_gather_scatter": score_rules(constant, truth),
    }
