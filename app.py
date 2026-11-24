"""Review screen for the explainable AML pipeline. Run with:  streamlit run app.py

Reads saved artifacts only (no live LLM calls). The optional "Regenerate note" button
is the one exception and needs OPENAI_API_KEY. Style follows the Project 1 app.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.audit_log import latest_decision_by_alert, read_decisions, record_decision
from src.totals import format_totals
from src.ui_components import (
    OK_COLOUR,
    REVIEW_COLOUR,
    SYNTHETIC_NOTE,
    badge,
    highlight_cited,
    kpi_card,
    show_table,
)
from src.ui_data import (
    cited_ids,
    load_bundle,
    load_json_if_present,
    load_queue,
    review_label,
)
from src.ui_graph import build_figure, graph_edges

FOLDER = config.ARTIFACTS_DIR
PAGES = ["Alert queue", "Alert detail", "Audit log", "Metrics"]
SORT_CHOICES = ["priority_rank", "score", "typology_hint", "review"]
SCORE_COLUMN = {"score": st.column_config.NumberColumn("score", format="%.3f")}

st.set_page_config(page_title="Explainable AML pipeline", layout="wide")


@st.cache_data(show_spinner="Loading alerts...")
def get_queue() -> pd.DataFrame:
    return load_queue(FOLDER)


def get_bundle(alert_id: str) -> dict:
    """Saved bundle, with a regenerated note on top if the analyst asked for one."""
    bundle = load_bundle(FOLDER, alert_id)
    regenerated = st.session_state.get("regenerated", {}).get(alert_id)
    if regenerated:
        bundle["record"] = regenerated
    return bundle


def selected_alert(queue: pd.DataFrame) -> str:
    ids = queue.sort_values("priority_rank")["alert_id"].tolist()
    current = st.session_state.get("alert_id", ids[0])
    chosen = st.sidebar.selectbox("Alert", ids, index=ids.index(current))
    st.session_state["alert_id"] = chosen
    return chosen


def page_queue(queue: pd.DataFrame) -> None:
    st.header("Alert queue")
    st.caption(
        "Priority is the detector score order. Typology is a hint, not a result."
    )
    flag = st.radio("Show", ["all", "needs review only", "clean only"], horizontal=True)
    hints = st.multiselect("Typology hint", sorted(queue["typology_hint"].unique()))
    sort = st.selectbox("Sort by", SORT_CHOICES)
    view = queue.copy()
    if flag != "all":
        wanted = review_label(flag == "needs review only")
        view = view[view["review"] == wanted]
    if hints:
        view = view[view["typology_hint"].isin(hints)]
    view = view.sort_values(sort, ascending=sort != "score")
    view["decision"] = view["alert_id"].map(latest_decision_by_alert()).fillna("")
    st.caption("%d of %d alerts" % (len(view), len(queue)))
    show_table(view, column_config=SCORE_COLUMN)


def show_status(record: dict) -> None:
    """Fact-check status of a note."""
    if record is None or record["note"] is None:
        st.warning("No valid note for this alert.")
    elif record["needs_review"]:
        st.markdown(badge(review_label(True), REVIEW_COLOUR), unsafe_allow_html=True)
        for item in record["violations"]:
            st.write("• " + item)
        st.caption("Fact check failed: the items above are not in the evidence.")
    else:
        st.markdown(badge(review_label(False), OK_COLOUR), unsafe_allow_html=True)
        st.caption("Every cited id, account and amount is in this alert's evidence.")


def show_note(bundle: dict) -> None:
    record = bundle["record"]
    show_status(record)
    note = record["note"] if record else None
    if not note:
        return
    st.markdown("**Typology hint:** %s (a hint, not a result)" % note["typology"])
    st.write(note["reason"])
    st.markdown("**Red flags**")
    for flag in note["red_flags"]:
        st.write("• " + flag)
    st.markdown("**Questions for the investigator**")
    for question in note["investigator_questions"]:
        st.write("• " + question)
    st.caption(
        "Cited: %d transactions, total %s (computed in code, per currency)"
        % (len(note["cited_txn_ids"]), format_totals(note["totals_by_currency"]))
    )


def regenerate_button(alert_id: str, bundle: dict) -> None:
    """Optional live call; the saved note stays if it fails."""
    if not st.button("Regenerate note (calls OpenAI)"):
        return
    from src.llm_client import make_client
    from src.run_step2 import note_for_alert

    try:
        client = make_client()
        record = note_for_alert(
            client, bundle["alert"], bundle["evidence"], config.UI_NOTE_VERSION
        )
    except Exception as error:  # no key or no network: say so, keep the saved note
        st.error("Could not regenerate: %s" % error)
        return
    st.session_state.setdefault("regenerated", {})[alert_id] = record
    st.rerun()


def page_detail(alert_id: str) -> None:
    bundle = get_bundle(alert_id)
    alert = bundle["alert"]
    st.header("%s  ·  score %.3f" % (alert_id, alert["score"]))
    st.caption(
        "Priority rank %d  ·  %s of all test transactions by score"
        % (alert["rank"], bundle["score_rank"])
    )
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Why the detector flagged it")
        for reason in bundle["reasons"]:
            st.write("• " + reason)
        st.subheader("Evidence")
        counts = bundle["evidence"]["counts"]
        st.caption(
            "%d linked transactions shown of %d found. Highlighted rows are cited "
            "by the note."
            % (counts["included"], counts["hop1_found"] + counts["hop2_found"])
        )
        show_table(highlight_cited(bundle["table"], cited_ids(bundle["record"])))
    with right:
        st.subheader("Case note")
        show_note(bundle)
        regenerate_button(alert_id, bundle)
        st.subheader("Network")
        edges = graph_edges(bundle["evidence"], cited_ids(bundle["record"]))
        st.plotly_chart(build_figure(edges), use_container_width=True)
        st.caption(
            "Accounts are nodes, transactions are edges. Red: the alert; "
            "orange: cited by the note. Hover an edge for amount, currency, time."
        )
    decision_form(alert_id, bundle)


def decision_form(alert_id: str, bundle: dict) -> None:
    """Escalate or close, with a comment, saved to the audit log."""
    st.divider()
    st.subheader("Your decision")
    show_status(bundle["record"])
    decision = st.radio(
        "Decision", list(config.DECISIONS), horizontal=True, key="d" + alert_id
    )
    comment = st.text_area("Comment (why)", key="c" + alert_id)
    if st.button("Save decision", key="s" + alert_id):
        record_decision(alert_id, decision, comment, config.UI_NOTE_VERSION)
        st.success("Saved %s for %s." % (decision, alert_id))


def page_audit_log() -> None:
    st.header("Audit log")
    log = read_decisions()
    if log:
        show_table(pd.DataFrame(log[::-1]))
    else:
        st.caption("No decisions yet. Decide on the Alert detail page.")


def number_cards(items: list) -> None:
    for column, (label, value, sub) in zip(st.columns(len(items)), items):
        with column:
            kpi_card(label, value, sub)


def detector_section(step1: dict) -> None:
    test = step1["test_metrics"]
    st.subheader("Detector (LightGBM, whole test period)")
    number_cards(
        [
            ("Minority F1", "%.3f" % test["minority_f1"], "matches the Phase E record"),
            ("Precision", "%.3f" % test["precision"], "at the validation threshold"),
            ("Recall", "%.3f" % test["recall"], "most laundering is missed"),
            ("PR-AUC", "%.3f" % test["pr_auc"], "a weak detector"),
            ("Precision@100", "%.2f" % test["precision_at_100"], "top 100 scores"),
        ]
    )
    share = 100 * step1["laundering_share_of_alerts"]
    st.caption("Of the top 200 alerts, %.1f%% are laundering." % share)


def notes_section(step2: dict) -> None:
    st.subheader("Case notes (prompt %s, 200 test alerts)" % step2["prompt_version"])
    typology = step2["typology"]
    number_cards(
        [
            ("JSON valid", "%d/%d" % (step2["json_valid"], step2["alerts"]),
             "one retry"),
            ("Typology accuracy", "%.2f" % typology["accuracy"],
             "%d mapped alerts" % typology["mapped_alerts"]),
            ("Notes needing review", str(step2["notes_needing_review"]),
             "%.1f%% invented a fact" % (100 * step2["invented_fact_rate_per_note"])),
            ("Cost", "$%.2f" % step2["cost_usd"],
             "%.1f s per note" % step2["mean_seconds_per_note"]),
        ]
    )  # fmt: skip


def page_metrics() -> None:
    st.header("Metrics")
    step1 = load_json_if_present(FOLDER / config.SUMMARY_FILE)
    step2 = load_json_if_present(FOLDER / config.EVAL_STEP2_FILE)
    if step1:
        detector_section(step1)
    if step2:
        notes_section(step2)
    st.subheader("Read these numbers with care")
    st.markdown(
        "- All data is IBM's **synthetic** LI-Small set; nothing here says how a real "
        "bank would fare.\n"
        "- The detector is weak; this pipeline is about explanation, not detection.\n"
        "- **ACH is a known shortcut in the IBM synthetic data**: the payment format "
        "is often the strongest reason, so a high score may reflect how the data was "
        "generated, not real-world suspicion.\n"
        "- The typology is a **hint**: about one in four is right, and the LLM never "
        'said "unclear" on a false alarm.\n'
        "- An earlier version let the LLM decide: escalate precision 0.27, the same as "
        "the alert base rate 0.26. The decision now belongs to the analyst.\n"
        "- The fact check shows a cited id, account or amount exists in the evidence, "
        "not that the story drawn from it is right."
    )


def main() -> None:
    st.sidebar.title("Explainable AML pipeline")
    page = st.sidebar.radio("Page", PAGES)
    queue = get_queue()
    alert_id = selected_alert(queue) if page == "Alert detail" else ""
    if page == "Alert queue":
        page_queue(queue)
    elif page == "Alert detail":
        page_detail(alert_id)
    elif page == "Audit log":
        page_audit_log()
    else:
        page_metrics()
    st.sidebar.caption(SYNTHETIC_NOTE)


main()
