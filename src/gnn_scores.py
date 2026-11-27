"""GNN second opinion: join the exported GNN scores to the alerts and their evidence.

`artifacts/gnn_scores.parquet` (written on Colab by src/gnn_export.py) holds one
laundering score per LI-Small validation and test transaction from aml-gnn-transfer's
fine-tuned PNA. The ids are IBM EdgeIDs, the pipeline's `transaction_id`, so the
join is a lookup. Training-period transactions (before GNN_SCORED_FROM) are not in
the file on purpose: the GNN learned their labels, so a score there would leak them.

LightGBM keeps the priority. The GNN score is shown beside it, never used to rank,
and never given to the note writer.

    python -m src.gnn_scores      # coverage check over the 200 alerts

Example:
    gnn = load_gnn_scores(config.ARTIFACTS_DIR)
    second_opinion_text(gnn, 3157594)   # "GNN second opinion: 1.00, high (...)"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd

from src import config
from src.case_note_input import evidence_transactions

HIGH, LOW, NOT_SCORED = "high", "low", "not scored"
AGREE, DISAGREE, NOT_AVAILABLE = "agree", "disagree", "n/a"
MAX_SHOWN_PERCENT = 99.99  # above this the header says "higher than >99.99%"
NEAR_TOP_PERCENT = 99.0  # from here two decimals, so 99.95 and 99.99 differ


class GnnScores(NamedTuple):
    by_id: pd.Series  # transaction_id -> score (validation and test periods)
    test_sorted: np.ndarray  # every test-period score, ascending, for percentiles


def load_gnn_scores(folder: Path) -> Optional[GnnScores]:
    """The exported scores, or None when the file is not there (UI works without)."""
    path = folder / config.GNN_SCORES_FILE
    if not path.exists():
        return None
    table = pd.read_parquet(path, columns=["transaction_id", "score", "period"])
    by_id = table.set_index("transaction_id")["score"].astype(float)
    test = np.sort(table.loc[table["period"] == "test", "score"].to_numpy())
    return GnnScores(by_id, test)


def score_of(gnn: GnnScores, transaction_id: int) -> Optional[float]:
    """The GNN score of one transaction, or None if it has none."""
    value = gnn.by_id.get(transaction_id)
    return None if value is None else float(value)


def band(score: Optional[float]) -> str:
    """ "high" from the validation-chosen threshold up, "low" below, or unscored."""
    if score is None:
        return NOT_SCORED
    return HIGH if score >= config.GNN_HIGH_THRESHOLD else LOW


def higher_than_percent(gnn: GnnScores, score: float) -> float:
    """Share (%) of test-period transactions the GNN scores strictly lower."""
    below = np.searchsorted(gnn.test_sorted, score, "left")
    return 100.0 * below / max(len(gnn.test_sorted), 1)


def percent_text(share: float) -> str:
    """ "28.6", "99.95", or ">99.99" (near 100 the decimals matter)."""
    if share > MAX_SHOWN_PERCENT:
        return ">%.2f" % MAX_SHOWN_PERCENT
    return ("%.2f" if share >= NEAR_TOP_PERCENT else "%.1f") % share


def second_opinion_text(gnn: Optional[GnnScores], transaction_id: int) -> str:
    """The alert header line."""
    score = score_of(gnn, transaction_id) if gnn is not None else None
    if score is None:
        return "GNN second opinion: not available"
    return "GNN second opinion: %.2f, %s (higher than %s%% of test transactions)" % (
        score, band(score), percent_text(higher_than_percent(gnn, score)),
    )  # fmt: skip


def agreement(lightgbm_high: bool, score: Optional[float]) -> str:
    """ "agree", "disagree" (one high, the other low) or "n/a" (no GNN score).

    LightGBM is high above its own validation threshold (every alert is), the GNN
    from GNN_HIGH_THRESHOLD; both thresholds were chosen on validation.
    """
    if score is None:
        return NOT_AVAILABLE
    return AGREE if lightgbm_high == (band(score) == HIGH) else DISAGREE


def disagreement_text(lightgbm_score: float, lightgbm_high: bool, score: float) -> str:
    """The badge text when the two models disagree."""
    return "LightGBM and GNN disagree: LightGBM %.2f (%s its threshold), GNN %.2f " \
        "(%s %.3f)" % (lightgbm_score, "above" if lightgbm_high else "below", score,
                       "above" if band(score) == HIGH else "below",
                       config.GNN_HIGH_THRESHOLD)  # fmt: skip


def evidence_scores(gnn: Optional[GnnScores], evidence: Dict) -> Dict[int, float]:
    """transaction_id -> score for the evidence rows that have one."""
    if gnn is None:
        return {}
    scores = {}
    for item in evidence_transactions(evidence):
        score = score_of(gnn, item["transaction_id"])
        if score is not None:
            scores[item["transaction_id"]] = score
    return scores


def coverage_problems(gnn: GnnScores, evidence: Dict) -> List[str]:
    """Rows that should have a score and do not (a wrong id mapping shows here)."""
    problems = []
    for item in evidence_transactions(evidence):
        due = item["time"] >= config.GNN_SCORED_FROM
        if due and score_of(gnn, item["transaction_id"]) is None:
            problems.append("txn %d at %s has no GNN score"
                            % (item["transaction_id"], item["time"]))  # fmt: skip
    return problems


def coverage_report(gnn: GnnScores, documents: List[Dict]) -> Dict:
    """Scored and unscored (training-period) rows over all alerts; problems listed."""
    rows = [t for d in documents for t in evidence_transactions(d)]
    scored = sum(score_of(gnn, t["transaction_id"]) is not None for t in rows)
    early = sum(t["time"] < config.GNN_SCORED_FROM for t in rows)
    problems = [p for d in documents for p in coverage_problems(gnn, d)]
    return {"alerts": len(documents), "evidence_rows": len(rows), "scored": scored,
            "training_period_unscored": early, "problems": problems}  # fmt: skip


def main() -> None:
    gnn = load_gnn_scores(config.ARTIFACTS_DIR)
    if gnn is None:
        raise SystemExit("No %s in artifacts/" % config.GNN_SCORES_FILE)
    folder = config.ARTIFACTS_DIR / config.EVIDENCE_FOLDER
    documents = [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]
    report = coverage_report(gnn, documents)
    print(json.dumps({**report, "problems": report["problems"][:5]}, indent=1))
    if report["problems"]:
        raise SystemExit("%d rows lack a GNN score" % len(report["problems"]))


if __name__ == "__main__":
    main()
