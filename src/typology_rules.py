"""A rule-based typology baseline: read the pattern off the evidence graph, no LLM.

It sees only what the student sees (the evidence transactions) and asks, around the
alert transaction S -> R, within `window_hours` of it:

- CYCLE: money from R comes back to S (a directed path R -> ... -> S);
- SCATTER_GATHER: one account pays at least `min_degree` accounts that all pay one
  other account (the alert is one of the scatter or gather legs);
- GATHER_SCATTER: S or R receives from at least `hub_min_payers` accounts and pays
  at least `min_degree` accounts;
- FAN_IN: R receives from at least `min_degree` accounts;
- FAN_OUT: S pays at least `min_degree` accounts;
- otherwise "unclear".

The first rule that holds wins, in that order. The three settings are chosen on
the dev set only (`tune`). It answers "is the typology this hard?": if these rules beat
the student, the student is not reading the graph.

Example:
    settings = RuleSettings(window_hours=24, min_degree=3, hub_min_payers=3)
    detect_typology(evidence, settings)   # "FAN_IN", ..., or "unclear"
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Set, Tuple

import networkx as nx

from src import config
from src.case_note_input import evidence_transactions

TIME_FORMAT = "%Y-%m-%d %H:%M"


@dataclass(frozen=True)
class RuleSettings:
    """How far in time to look, and how many counterparties make a fan or a hub."""

    window_hours: float
    min_degree: int
    hub_min_payers: int


def window_graph(evidence: Dict, window_hours: float) -> nx.MultiDiGraph:
    """Accounts as nodes, evidence transactions near the alert in time as edges."""
    transactions = evidence_transactions(evidence)
    alert_time = datetime.strptime(transactions[0]["time"], TIME_FORMAT)
    graph = nx.MultiDiGraph()
    for item in transactions:
        hours = abs(datetime.strptime(item["time"], TIME_FORMAT) - alert_time)
        if hours.total_seconds() <= window_hours * 3600:
            graph.add_edge(item["from_account"], item["to_account"])
    return graph


def payees(graph: nx.MultiDiGraph, account: str) -> Set[str]:
    """Distinct accounts this account paid (self-payments left out)."""
    return {other for other in graph.successors(account) if other != account}


def payers(graph: nx.MultiDiGraph, account: str) -> Set[str]:
    """Distinct accounts that paid this account (self-payments left out)."""
    return {other for other in graph.predecessors(account) if other != account}


def has_cycle(graph: nx.MultiDiGraph, sender: str, receiver: str) -> bool:
    """True if money paid to the receiver can flow back to the sender."""
    if sender == receiver:
        return False
    return nx.has_path(graph, receiver, sender)


def is_scatter_gather(
    graph: nx.MultiDiGraph, sender: str, receiver: str, k: int
) -> bool:
    """True if the alert is a leg of source -> many middles -> one sink."""
    for source in [sender] + sorted(payers(graph, sender)):
        middles = payees(graph, source)
        if len(middles) < k or (source != sender and sender not in middles):
            continue
        sinks: Dict[str, int] = {}
        for middle in middles:
            for sink in payees(graph, middle) - {source}:
                sinks[sink] = sinks.get(sink, 0) + 1
        if any(count >= k for count in sinks.values()):
            return True
    return False


def is_hub(graph: nx.MultiDiGraph, account: str, settings: RuleSettings) -> bool:
    """True if the account both gathers from and scatters to enough accounts."""
    return (
        len(payers(graph, account)) >= settings.hub_min_payers
        and len(payees(graph, account)) >= settings.min_degree
    )


def detect_typology(evidence: Dict, settings: RuleSettings) -> str:
    """The first rule that holds for this evidence, else "unclear"."""
    graph = window_graph(evidence, settings.window_hours)
    alert = evidence["alert_transaction"]
    sender, receiver, k = (
        alert["from_account"],
        alert["to_account"],
        settings.min_degree,
    )
    if has_cycle(graph, sender, receiver):
        return "CYCLE"
    if is_scatter_gather(graph, sender, receiver, k):
        return "SCATTER_GATHER"
    if is_hub(graph, sender, settings) or is_hub(graph, receiver, settings):
        return "GATHER_SCATTER"
    if len(payers(graph, receiver)) >= k:
        return "FAN_IN"
    if len(payees(graph, sender)) >= k:
        return "FAN_OUT"
    return config.UNCLEAR


def score_rules(predictions: Dict[str, str], truth: List[Dict]) -> Dict:
    """Right typology on visible laundering, and "unclear" on false alarms."""
    visible = [t for t in truth if t["is_laundering"] and t["pattern_visible"]]
    false_alarms = [t for t in truth if not t["is_laundering"]]
    return {
        "typology_right_visible": sum(
            predictions[t["alert_id"]] == t["pattern_type"] for t in visible
        ),
        "visible": len(visible),
        "unclear_false_alarms": sum(
            predictions[t["alert_id"]] == config.UNCLEAR for t in false_alarms
        ),
        "false_alarms": len(false_alarms),
    }


def tune(
    evidence: Dict[str, Dict], truth: List[Dict], grid: List[RuleSettings]
) -> Tuple[RuleSettings, Dict]:
    """Settings with the most right typologies; ties go to more unclear false alarms."""
    best_key, best = None, None
    for settings in grid:
        predicted = {i: detect_typology(e, settings) for i, e in evidence.items()}
        result = score_rules(predicted, truth)
        key = (result["typology_right_visible"], result["unclear_false_alarms"])
        if best_key is None or key > best_key:
            best_key, best = key, (settings, result)
    return best
