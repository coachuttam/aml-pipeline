"""Build the model's input and the JSON schema of its answer.

The input holds the score, plain-English reasons and evidence transactions only (prompt
v4 gets the evidence transactions only). No
label and no pattern type exists in these files (AGENTS.md rule 2).
"""

from __future__ import annotations

import json
from typing import Dict, List

from src import config
from src.reasons import plain_reasons


def load_prompt(version: str) -> str:
    """The versioned system prompt, for example version "v2"."""
    path = config.PROMPTS_DIR / ("case_note_%s.txt" % version)
    return path.read_text()


def evidence_transactions(evidence: Dict) -> List[Dict]:
    """The alert transaction followed by its linked transactions."""
    return [evidence["alert_transaction"]] + evidence["linked_transactions"]


def build_user_message(alert: Dict, evidence: Dict) -> str:
    """Alert id, score, reasons and every evidence transaction as JSON."""
    payload = {
        "alert_id": alert["alert_id"],
        "alert_transaction_id": evidence["alert_transaction"]["transaction_id"],
        "model_score": round(float(alert["score"]), 4),
        "reasons": plain_reasons(alert),
        "evidence_note": "%d linked transactions were found; %d are shown."
        % (evidence["counts"]["hop1_found"] + evidence["counts"]["hop2_found"],
           len(evidence["linked_transactions"])),
        "transactions": evidence_transactions(evidence),
    }  # fmt: skip
    return json.dumps(payload, indent=1)


def build_blind_user_message(alert: Dict, evidence: Dict) -> str:
    """Evidence only: the alert transaction id and the transactions, nothing else.

    No model score, no SHAP reasons, no alert id and no count of linked transactions
    found: each of these differs between detectors and datasets (a student that sees
    them learns the dataset, not the transactions). The reasons stay in the UI.
    """
    payload = {
        "alert_transaction_id": evidence["alert_transaction"]["transaction_id"],
        "transactions": evidence_transactions(evidence),
    }
    return json.dumps(payload, indent=1)


def user_message_for(version: str, alert: Dict, evidence: Dict) -> str:
    """The input a prompt version expects (blind versions get the evidence only)."""
    if version in config.BLIND_PROMPT_VERSIONS:
        return build_blind_user_message(alert, evidence)
    return build_user_message(alert, evidence)


def response_format() -> Dict:
    """OpenAI structured-output schema for one case note."""
    text_list = {"type": "array", "items": {"type": "string"}}
    properties = {
        "typology": {
            "type": "string",
            "enum": list(config.TYPOLOGIES) + [config.UNCLEAR],
        },
        "key_accounts": text_list,
        "cited_txn_ids": {"type": "array", "items": {"type": "integer"}},
        "red_flags": text_list,
        "investigator_questions": text_list,
        "reason": {"type": "string"},
    }
    schema = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {"name": "case_note", "strict": True, "schema": schema},
    }
