"""The decision form and the goAML-style draft section of the alert detail page.

The analyst escalates or closes an alert; when escalating they may confirm the
suggested typology. Only an alert whose latest decision is "escalate" gets a
"Draft goAML report" button: it builds the draft (src/goaml_draft.py), checks it
against our own schema, lists the missing fields, shows the XML and offers it for
download. Making a draft is logged; nothing is ever filed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st

from src import config
from src.audit_log import (
    current_escalation,
    latest_decisions,
    record_decision,
    record_draft,
)
from src.goaml_check import missing_fields, validation_errors
from src.goaml_draft import draft_xml

DRAFT_LABEL = "goAML-style draft, not a certified UAE FIU schema. Never filed."


def suggested_typology(bundle: dict) -> Optional[str]:
    """The note's typology when it names one (not "unclear")."""
    note = (bundle["record"] or {}).get("note")
    if not note or note["typology"] == config.UNCLEAR:
        return None
    return note["typology"]


def decision_form(alert_id: str, bundle: dict, show_status) -> None:
    """Escalate or close, with a comment and an optional typology confirmation."""
    st.divider()
    st.subheader("Your decision")
    show_status(bundle["record"])
    decision = st.radio(
        "Decision", list(config.DECISIONS), horizontal=True, key="d" + alert_id
    )
    comment = st.text_area("Comment (why)", key="c" + alert_id)
    typology = suggested_typology(bundle)
    confirmed = typology and decision == "escalate" and st.checkbox(
        "I confirm the suggested typology %s (it then goes into the goAML draft)"
        % typology, key="t" + alert_id,
    )  # fmt: skip
    if st.button("Save decision", key="s" + alert_id):
        version = (bundle["record"] or {}).get("note_version", config.UI_NOTE_VERSION)
        record_decision(alert_id, decision, comment, version,
                        typology_confirmed=typology if confirmed else "")  # fmt: skip
        st.success("Saved %s for %s." % (decision, alert_id))
    saved = latest_decisions().get(alert_id)
    if saved:
        st.caption(saved_state_text(saved))
    goaml_section(alert_id, bundle)


def saved_state_text(event: dict) -> str:
    """The saved decision as the log has it, with its typology confirmation."""
    typology = event.get("typology_confirmed")
    confirmed = ("typology confirmed: %s" % typology) if typology else (
        "typology not confirmed")  # fmt: skip
    return "Saved decision: %s at %s · %s" % (
        event["decision"], event["time"].replace("T", " "), confirmed,
    )  # fmt: skip


def make_draft(alert_id: str, bundle: dict, escalation: dict) -> dict:
    """Build, check and log one draft; the result is kept for preview."""
    submitted = datetime.now().isoformat(timespec="seconds")
    xml = draft_xml(bundle["alert"], bundle["evidence"], bundle["record"],
                    escalation, submitted)  # fmt: skip
    gaps = missing_fields(xml)
    typology = escalation.get("typology_confirmed", "")
    record_draft(alert_id, sum(g["count"] for g in gaps), typology)
    return {"xml": xml, "errors": validation_errors(xml), "missing": gaps}


def goaml_section(alert_id: str, bundle: dict) -> None:
    """Draft button, schema result, missing fields, XML preview and download."""
    escalation = current_escalation(alert_id)
    if escalation is None:
        st.caption("A goAML-style draft can be made once the alert is escalated.")
        return
    st.subheader("goAML-style STR draft")
    st.caption(DRAFT_LABEL)
    drafts = st.session_state.setdefault("goaml_drafts", {})
    if st.button("Draft goAML report", key="g" + alert_id):
        drafts[alert_id] = make_draft(alert_id, bundle, escalation)
    draft = drafts.get(alert_id)
    if draft:
        show_draft(alert_id, draft, bundle)


def show_draft(alert_id: str, draft: dict, bundle: dict) -> None:
    if draft["errors"]:
        st.error("Does not fit our draft schema:\n\n" + "\n\n".join(draft["errors"]))
    else:
        st.success("Fits our goAML-style draft schema (not a UAE FIU certification).")
    if bundle["record"] and bundle["record"].get("needs_review"):
        st.warning("The case note failed the fact check: check the reason text.")
    st.markdown("**Missing fields (%d kinds) — fill in before filing**"
                % len(draft["missing"]))  # fmt: skip
    table = [{"missing": g["field"], "times": g["count"],
              "where": ", ".join(g["where"])} for g in draft["missing"]]  # fmt: skip
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
    st.download_button("Download XML (draft)", draft["xml"],
                       file_name="goaml_style_draft_%s.xml" % alert_id,
                       mime="application/xml", key="x" + alert_id)  # fmt: skip
    with st.expander("Preview XML"):
        st.code(draft["xml"].decode("utf-8"), language="xml")
