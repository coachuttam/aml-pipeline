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
    """A styled table with the rows the note cites highlighted."""

    def colour(row: pd.Series) -> List[str]:
        cited_row = row["transaction_id"] in cited
        return [
            "background-color: %s; color: black" % config.CITED_COLOUR
            if cited_row
            else ""
        ] * len(row)

    return table.style.apply(colour, axis=1)
