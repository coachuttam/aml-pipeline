"""An interactive network of accounts (nodes) and transactions (edges) for one alert.

Hovering an edge shows the transaction's amount, currency, time and GNN score.
Two colourings: by note citations (the alert red, cited orange, the rest grey) or by
GNN score (light to dark, thicker from the validation threshold up, grey dashed
where there is no score; the alert keeps a red outline), with a legend.

Example:
    edges = graph_edges(evidence, cited=[5, 7], gnn={5: 0.91})
    figure = build_figure(edges, colour_by=GNN)       # a plotly figure
"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

from src import config
from src.case_note_input import evidence_transactions
from src.totals import currency_code

HOVER_POINTS_PER_EDGE = 3  # invisible points along an edge so hovering is easy
CITATIONS, GNN = "note citations", "GNN score"  # the two edge colourings
ALERT_OUTLINE_EXTRA = 4  # the alert's red outline is this much wider than its line


class Edge(NamedTuple):
    sender: str
    receiver: str
    transaction_id: int
    is_alert: bool
    is_cited: bool
    hover: str
    gnn_score: Optional[float] = None


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


def gnn_hover(score: Optional[float]) -> str:
    """ "GNN 0.81", or why there is no score."""
    if score is None:
        return "GNN: not scored (training period)"
    return "GNN %.2f" % score


def graph_edges(
    evidence: Dict, cited: List[int], gnn: Optional[Dict[int, float]] = None
) -> List[Edge]:
    """One edge per transaction, flagged as alert and/or cited, with its GNN score."""
    alert_id = evidence["alert_transaction"]["transaction_id"]
    wanted: Set[int] = set(cited)
    edges = []
    for t in evidence_transactions(evidence):
        score = gnn.get(t["transaction_id"]) if gnn is not None else None
        hover = hover_text(t)
        if gnn is not None:
            hover += "<br>" + gnn_hover(score)
        edges.append(Edge(t["from_account"], t["to_account"], t["transaction_id"],
                          t["transaction_id"] == alert_id,
                          t["transaction_id"] in wanted, hover, score))  # fmt: skip
    return edges


def edge_colour(edge: Edge) -> str:
    if edge.is_alert:
        return config.ALERT_EDGE_COLOUR
    return config.CITED_EDGE_COLOUR if edge.is_cited else config.OTHER_EDGE_COLOUR


def blend(low: str, high: str, share: float) -> str:
    """The colour `share` of the way from `low` to `high` (hex colours)."""
    a = [int(low[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(high[i : i + 2], 16) for i in (1, 3, 5)]
    share = min(max(share, 0.0), 1.0)
    return "#" + "".join("%02x" % round(x + (y - x) * share) for x, y in zip(a, b))


def gnn_style(edge: Edge) -> Tuple[str, int, str]:
    """Colour, width and dash of an edge coloured by GNN score."""
    if edge.gnn_score is None:
        return config.UNSCORED_EDGE_COLOUR, config.GNN_EDGE_WIDTH, "dash"
    colour = blend(config.GNN_LOW_COLOUR, config.GNN_HIGH_COLOUR, edge.gnn_score)
    high = edge.gnn_score >= config.GNN_HIGH_THRESHOLD
    width = config.GNN_HIGH_EDGE_WIDTH if high else config.GNN_EDGE_WIDTH
    return colour, width, "solid"


def edge_style(edge: Edge, colour_by: str) -> Tuple[str, int, str]:
    """Colour, width and dash in either colouring."""
    if colour_by == GNN:
        return gnn_style(edge)
    width = 4 if edge.is_alert else (3 if edge.is_cited else 1)
    return edge_colour(edge), width, "solid"


def layout(edges: List[Edge]) -> Dict[str, Any]:
    """Fixed account positions (same seed -> same picture)."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_edges_from((e.sender, e.receiver) for e in edges)
    return nx.spring_layout(graph, seed=0)


def points(start: float, end: float) -> List[float]:
    """Both ends plus evenly spaced hover points along one edge."""
    steps = HOVER_POINTS_PER_EDGE + 1
    return [start + (end - start) * k / steps for k in range(steps + 1)]


def line_trace(edge: Edge, positions: Dict[str, Any], style: Tuple[str, int, str]):
    import plotly.graph_objects as go

    (x0, y0), (x1, y1) = positions[edge.sender], positions[edge.receiver]
    colour, width, dash = style
    return go.Scatter(
        x=points(x0, x1), y=points(y0, y1), mode="lines+markers",
        line={"color": colour, "width": width, "dash": dash},
        marker={"size": 6, "opacity": 0},
        text=[edge.hover] * (HOVER_POINTS_PER_EDGE + 2), hoverinfo="text",
        showlegend=False,
    )  # fmt: skip


def edge_traces(
    edges: List[Edge], positions: Dict[str, Any], colour_by: str = CITATIONS
) -> List[Any]:
    """Lines (alert last, so it is on top) plus invisible hover points."""
    traces = []
    for edge in sorted(edges, key=lambda e: (e.is_alert, e.is_cited)):
        style = edge_style(edge, colour_by)
        if colour_by == GNN and edge.is_alert:  # red outline under its GNN colour
            outline = (
                config.ALERT_EDGE_COLOUR,
                style[1] + ALERT_OUTLINE_EXTRA,
                "solid",
            )
            traces.append(line_trace(edge, positions, outline))
        traces.append(line_trace(edge, positions, style))
    return traces


def legend_line(name: str, colour: str, dash: str = "solid") -> Any:
    """A legend-only entry (no data drawn)."""
    import plotly.graph_objects as go

    return go.Scatter(x=[None], y=[None], mode="lines", name=name,
                      line={"color": colour, "width": 3, "dash": dash})  # fmt: skip


def gnn_legend() -> List[Any]:
    """Colour bar 0-1 with the validation threshold, plus alert and unscored lines."""
    import plotly.graph_objects as go

    threshold = config.GNN_HIGH_THRESHOLD
    bar = go.Scatter(
        x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
        marker={"colorscale": [[0, config.GNN_LOW_COLOUR], [1, config.GNN_HIGH_COLOUR]],
                "cmin": 0, "cmax": 1, "color": [0], "showscale": True,
                "colorbar": {"title": {"text": "GNN laundering score", "side": "right"},
                             "tickvals": [0, threshold, 1],
                             "ticktext": ["0", "%.3f high (threshold from validation)"
                                          % threshold, "1"],
                             "thickness": 12, "len": 0.8}},
    )  # fmt: skip
    return [bar, legend_line("alert transaction (outline)", config.ALERT_EDGE_COLOUR),
            legend_line("not scored: GNN training period",
                        config.UNSCORED_EDGE_COLOUR, "dash")]  # fmt: skip


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


def padded_range(values: List[float]) -> List[float]:
    """Axis limits with room on both sides so long account labels are not clipped."""
    low, high = min(values), max(values)
    pad = (high - low or 1.0) * config.GRAPH_PADDING
    return [low - pad, high + pad]


def build_figure(edges: List[Edge], colour_by: str = CITATIONS) -> Any:
    """The interactive figure: edges first, account labels on top, GNN legend."""
    import plotly.graph_objects as go

    positions = layout(edges)
    traces = edge_traces(edges, positions, colour_by) + [node_trace(positions)]
    if colour_by == GNN:
        traces += gnn_legend()
    figure = go.Figure(traces)
    xs = [x for x, _ in positions.values()]
    ys = [y for _, y in positions.values()]
    figure.update_layout(
        height=config.GRAPH_HEIGHT_PIXELS,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        xaxis={"visible": False, "range": padded_range(xs)},
        yaxis={"visible": False, "range": padded_range(ys)},
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": -0.02, "font": {"size": 10}},
    )
    return figure
