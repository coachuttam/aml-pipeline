"""Small Streamlit pieces copied in style from the Project 1 app
(~/aml-alert-investigator/app.py and evidence_view.py; read only, not edited):
KPI card, pill badge, full-width table.
"""

from __future__ import annotations

from typing import List

import pandas as pd
import streamlit as st

from src import config

SYNTHETIC_NOTE = "Synthetic data. Prototype, not certified for compliance."
REVIEW_COLOUR = "rgba(210,153,34,0.22)"
OK_COLOUR = "rgba(46,160,67,0.18)"


def kpi_card(label: str, value: str, sub: str) -> None:
    """A bordered card: small label, big value, one line of context."""
    st.markdown(
        '<div style="border:1px solid rgba(128,128,128,0.25);'
        "border-radius:10px;padding:14px;"
        f'min-height:132px;"><div style="font-size:0.78rem;opacity:0.7;">{label}</div>'
        f'<div style="font-size:1.45rem;font-weight:650;">{value}</div>'
        f'<div style="font-size:0.8rem;opacity:0.8;">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def badge(text: str, colour: str) -> str:
    """A pill, as HTML."""
    style = (
        f"background:{colour};padding:2px 10px;border-radius:999px;"
        "font-size:0.7rem;font-weight:600;"
    )
    return f'<span style="{style}">{text}</span>'


def show_table(table: pd.DataFrame, **options) -> None:
    """A full-width table; newer Streamlit says width="stretch"."""
    try:
        st.dataframe(table, width="stretch", hide_index=True, **options)
    except (TypeError, ValueError):
        st.dataframe(table, use_container_width=True, hide_index=True, **options)


def highlight_cited(table: pd.DataFrame, cited: List[int]):
    """A styled table: cited rows highlighted, amounts to 2 decimals."""

    def colour(row: pd.Series) -> List[str]:
        cited_row = row["transaction_id"] in cited
        return [
            "background-color: %s; color: black" % config.CITED_COLOUR
            if cited_row
            else ""
        ] * len(row)

    return table.style.apply(colour, axis=1).format({"amount_paid": "{:,.2f}"})


def number_cards(items: list) -> None:
    """A row of KPI cards: (label, value, sub-line) each."""
    for column, (label, value, sub) in zip(st.columns(len(items)), items):
        with column:
            kpi_card(label, value, sub)


def student_section(report: dict) -> None:
    """The final test of the V2 student (run once), beside V1 prompt v3."""
    student, v1 = report["student"], report["v1_prompt_v3"]
    st.subheader("Case notes: V2 student (typology2), final test run once")

    def share(scores: dict, key: str) -> str:
        return "%d/%d" % (scores[key]["count"], scores[key]["of"])

    number_cards(
        [
            ("Typology right", share(student, "typology_accuracy_visible"),
             "visible patterns; V1 v3 " + share(v1, "typology_accuracy_visible")),
            ('"unclear" on false alarms', share(student, "unclear_rate_false_alarms"),
             "V1 v3 " + share(v1, "unclear_rate_false_alarms")),
            ("Invented facts", share(student, "invented_facts_per_claim"),
             "per claim; V1 v3 " + share(v1, "invented_facts_per_claim")),
            ("JSON valid", share(student, "json_validity"), "200 test alerts"),
        ]
    )  # fmt: skip


def network_section(bundle: dict) -> None:
    """The evidence network, coloured by note citations or by GNN score."""
    from src.ui_data import cited_ids
    from src.ui_graph import CITATIONS, GNN, build_figure, graph_edges

    st.subheader("Network")
    choices = [CITATIONS, GNN] if bundle["gnn"] is not None else [CITATIONS]
    colour_by = st.radio("Colour edges by", choices, horizontal=True)
    edges = graph_edges(bundle["evidence"], cited_ids(bundle["record"]), bundle["gnn"])
    st.plotly_chart(build_figure(edges, colour_by), use_container_width=True)
    if colour_by == GNN:
        st.caption(
            "Light to dark: GNN laundering score 0-1; thicker from %.3f (threshold "
            "chosen on validation). Grey dashed: not scored, because the transaction "
            "is in the GNN's training period and a score would reflect labels it "
            "learned. Red outline: the alert. A second opinion only."
            % config.GNN_HIGH_THRESHOLD
        )
    else:
        st.caption(
            "Accounts are nodes, transactions are edges. Red: the alert; "
            "orange: cited by the note. Hover an edge for amount, currency, time."
        )
