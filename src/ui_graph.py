"""An interactive network of accounts (nodes) and transactions (edges) for one alert.

Hovering an edge shows the transaction's amount, currency and time. The alert
transaction is red, transactions the note cites are orange, the rest grey.

Example:
    edges = graph_edges(evidence, cited=[5, 7])
    figure = build_figure(edges)       # a plotly figure
"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Set

from src import config
from src.case_note_input import evidence_transactions
from src.totals import currency_code

HOVER_POINTS_PER_EDGE = 3  # invisible points along an edge so hovering is easy


class Edge(NamedTuple):
    sender: str
    receiver: str
    transaction_id: int
    is_alert: bool
    is_cited: bool
    hover: str


def hover_text(item: Dict) -> str:
    """ "txn 5: 100.00 USD at 2022-09-09 00:03, A -> B"."""
    return "txn %d: %s %s at %s<br>%s → %s" % (
        item["transaction_id"],
        "{:,.2f}".format(item["amount_paid"]),
        currency_code(item["payment_currency"]),
        item["time"],
        item["from_account"],
        item["to_account"],
    )


def graph_edges(evidence: Dict, cited: List[int]) -> List[Edge]:
    """One edge per transaction, flagged as alert and/or cited."""
    alert_id = evidence["alert_transaction"]["transaction_id"]
    wanted: Set[int] = set(cited)
    return [
        Edge(
            t["from_account"],
            t["to_account"],
            t["transaction_id"],
            t["transaction_id"] == alert_id,
            t["transaction_id"] in wanted,
            hover_text(t),
        )
        for t in evidence_transactions(evidence)
    ]


def edge_colour(edge: Edge) -> str:
    if edge.is_alert:
        return config.ALERT_EDGE_COLOUR
    return config.CITED_EDGE_COLOUR if edge.is_cited else config.OTHER_EDGE_COLOUR


def layout(edges: List[Edge]) -> Dict[str, Any]:
    """Fixed account positions (same seed -> same picture)."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_edges_from((e.sender, e.receiver) for e in edges)
    return nx.spring_layout(graph, seed=0)


def edge_traces(edges: List[Edge], positions: Dict[str, Any]) -> List[Any]:
    """Lines (alert last, so it is on top) plus invisible hover points."""
    import plotly.graph_objects as go

    traces = []
    for edge in sorted(edges, key=lambda e: (e.is_alert, e.is_cited)):
        (x0, y0), (x1, y1) = positions[edge.sender], positions[edge.receiver]
        width = 4 if edge.is_alert else (3 if edge.is_cited else 1)
        traces.append(
            go.Scatter(
                x=[
                    x0 + (x1 - x0) * k / (HOVER_POINTS_PER_EDGE + 1)
                    for k in range(HOVER_POINTS_PER_EDGE + 2)
                ],
                y=[
                    y0 + (y1 - y0) * k / (HOVER_POINTS_PER_EDGE + 1)
                    for k in range(HOVER_POINTS_PER_EDGE + 2)
                ],
                mode="lines+markers",
                line={"color": edge_colour(edge), "width": width},
                marker={"size": 6, "opacity": 0},
                text=[edge.hover] * (HOVER_POINTS_PER_EDGE + 2),
                hoverinfo="text",
                showlegend=False,
            )
        )
    return traces


def node_trace(positions: Dict[str, Any]) -> Any:
    import plotly.graph_objects as go

    names = list(positions)
    return go.Scatter(
        x=[positions[n][0] for n in names],
        y=[positions[n][1] for n in names],
        mode="markers+text",
        text=names,
        textposition="top center",
        textfont={"size": 9},
        marker={"size": 9, "color": config.NODE_COLOUR},
        hoverinfo="text",
        hovertext=names,
        showlegend=False,
    )


def build_figure(edges: List[Edge]) -> Any:
    """The interactive figure: edges first, account labels on top."""
    import plotly.graph_objects as go

    positions = layout(edges)
    figure = go.Figure(edge_traces(edges, positions) + [node_trace(positions)])
    figure.update_layout(
        height=config.GRAPH_HEIGHT_PIXELS,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        xaxis={"visible": False},
        yaxis={"visible": False},
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure
