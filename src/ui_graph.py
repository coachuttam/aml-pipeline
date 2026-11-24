"""A small network of accounts (nodes) and transactions (edges) for one alert.

Example:
    edges = graph_edges(evidence)          # (from, to, transaction_id, is_alert)
    figure = draw_graph(edges)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src import config
from src.case_note_input import evidence_transactions

Edge = Tuple[str, str, int, bool]


def graph_edges(evidence: Dict) -> List[Edge]:
    """One edge per transaction; the alert transaction is marked True."""
    alert_id = evidence["alert_transaction"]["transaction_id"]
    return [
        (
            t["from_account"],
            t["to_account"],
            t["transaction_id"],
            t["transaction_id"] == alert_id,
        )
        for t in evidence_transactions(evidence)
    ]


def draw_graph(edges: List[Edge]) -> Any:
    """A matplotlib figure with the alert transaction in red."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    graph = nx.MultiDiGraph()
    for sender, receiver, transaction_id, _ in edges:
        graph.add_edge(sender, receiver, key=transaction_id)
    positions = nx.spring_layout(graph, seed=0)
    figure, axes = plt.subplots(figsize=config.GRAPH_FIGURE_SIZE)
    nx.draw_networkx_nodes(
        graph, positions, node_size=40, node_color="#4c78a8", ax=axes
    )
    normal = [(s, r) for s, r, _, is_alert in edges if not is_alert]
    alert = [(s, r) for s, r, _, is_alert in edges if is_alert]
    nx.draw_networkx_edges(
        graph, positions, edgelist=normal, edge_color="#aaaaaa", arrowsize=6, ax=axes
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=alert,
        edge_color=config.ALERT_EDGE_COLOUR,
        width=2.5,
        ax=axes,
    )
    axes.set_axis_off()
    return figure
