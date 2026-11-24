"""Totals of the cited transactions, one per currency (never summed across currencies).

Example:
    totals_by_currency([1, 2], by_id)    # {"USD": 150.5, "EUR": 20.0}
    format_totals({"USD": 150.5, "EUR": 20.0})   # "USD 150.50 · EUR 20.00"
"""

from __future__ import annotations

from typing import Dict, List

from src import config


def currency_code(name: str) -> str:
    """ "US Dollar" -> "USD"; an unknown name is kept as it is."""
    return config.CURRENCY_CODES.get(name, name)


def totals_by_currency(
    cited_ids: List[int], by_id: Dict[int, Dict]
) -> Dict[str, float]:
    """Sum of amount_paid per currency over the cited transactions that exist."""
    totals: Dict[str, float] = {}
    for transaction_id in cited_ids:
        if transaction_id in by_id:
            item = by_id[transaction_id]
            code = currency_code(item["payment_currency"])
            totals[code] = round(totals.get(code, 0.0) + item["amount_paid"], 2)
    return totals


def format_totals(totals: Dict[str, float]) -> str:
    """ "USD 150.50 · EUR 20.00", or "none" when nothing is cited."""
    if not totals:
        return "none"
    return " · ".join(
        "%s %s" % (code, "{:,.2f}".format(v)) for code, v in totals.items()
    )
