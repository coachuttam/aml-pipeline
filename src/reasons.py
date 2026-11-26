"""Turn a SHAP reason (feature, value, push) into one plain-English sentence.

Example:
    plain_reason("sender_sent_7d", 57.0, 10.3, alert_row)
    -> "The sender made 57 payments in the last 7 days (raises the score by 10.3)."
"""

from __future__ import annotations

import json
from typing import Dict, List

from src import config

# Code features are shown as the alert's own text value, not the meaningless code.
TEMPLATES: Dict[str, str] = {
    "amount_vs_receiver_mean": (
        "This amount is {value:.1f} times the receiver's usual amount received"
    ),
    "amount_vs_sender_mean": (
        "This amount is {value:.1f} times the sender's usual amount sent"
    ),
    "is_cross_bank": "The payment {cross} between two different banks",
    "is_first_time_pair": "The sender and receiver {first} paid each other before",
    "log_amount_paid": (
        "The amount paid is large or small for this data (log amount {value:.1f})"
    ),
    "receiver_in_degree": "The receiver has had {value:.0f} different senders so far",
    "receiver_mean_amount_received": (
        "The receiver's average amount received is {value:.2f}"
    ),
    "receiver_new_senders_24h": (
        "The receiver got payments from {value:.0f} new senders in the last 24 hours"
    ),
    "receiver_new_senders_7d": (
        "The receiver got payments from {value:.0f} new senders in the last 7 days"
    ),
    "receiver_out_degree": (
        "The receiver has paid {value:.0f} different accounts so far"
    ),
    "receiver_received_1h": "The receiver got {value:.0f} payments in the last hour",
    "receiver_received_24h": (
        "The receiver got {value:.0f} payments in the last 24 hours"
    ),
    "receiver_received_7d": "The receiver got {value:.0f} payments in the last 7 days",
    "sender_in_degree": (
        "The sender has had {value:.0f} different senders pay it so far"
    ),
    "sender_in_out_ratio": (
        "The sender's ratio of incoming to outgoing partners is {value:.2f}"
    ),
    "sender_new_partners_24h": (
        "The sender paid {value:.0f} new accounts in the last 24 hours"
    ),
    "sender_new_partners_7d": (
        "The sender paid {value:.0f} new accounts in the last 7 days"
    ),
    "sender_out_degree": "The sender has paid {value:.0f} different accounts so far",
    "sender_sent_1h": "The sender made {value:.0f} payments in the last hour",
    "sender_sent_24h": "The sender made {value:.0f} payments in the last 24 hours",
    "sender_sent_7d": "The sender made {value:.0f} payments in the last 7 days",
}


def reason_text(feature: str, value: float, alert: Dict) -> str:
    """The reason as plain words, without the size of its push."""
    if feature == "payment_format_code":
        text = "The payment format is %s" % alert["payment_format"]
    elif feature == "payment_currency_code":
        text = "The payment currency is %s" % alert["payment_currency"]
    elif feature in TEMPLATES:
        text = TEMPLATES[feature].format(
            value=value,
            cross="is" if value else "is not",
            first="have not" if value else "have",
        )
    else:
        text = "%s is %s" % (feature.replace("_", " "), value)
    return text


def plain_reason(feature: str, value: float, push: float, alert: Dict) -> str:
    """One sentence for one reason; the push is how far it raised the model score.

    This exact text is the LLM's input (changing it would change every cache key).
    """
    direction = "raises" if push >= 0 else "lowers"
    text = reason_text(feature, value, alert)
    return "%s (%s the score by %.1f)." % (text, direction, abs(push))


def strength_label(share: float) -> str:
    """ "strong", "moderate" or "weak" from a reason's share of the top-5 push."""
    if share >= config.STRONG_REASON_SHARE:
        return "strong"
    return "moderate" if share >= config.MODERATE_REASON_SHARE else "weak"


def display_reasons(alert: Dict) -> List[Dict]:
    """For the UI: text, strength label, % share of the top-5 push, direction."""
    reasons = json.loads(alert["reasons_json"])
    total = sum(abs(r["shap"]) for r in reasons) or 1.0
    shown = []
    for r in reasons:
        share = abs(r["shap"]) / total
        shown.append(
            {
                "text": reason_text(r["feature"], r["value"], alert),
                "strength": strength_label(share),
                "share_percent": round(100 * share),
                "direction": "raises" if r["shap"] >= 0 else "lowers",
            }
        )
    return shown


def plain_reasons(alert: Dict) -> List[str]:
    """The alert's top reasons, in order, as sentences."""
    return [
        plain_reason(r["feature"], r["value"], r["shap"], alert)
        for r in json.loads(alert["reasons_json"])
    ]
