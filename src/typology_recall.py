"""Recall per typology on the LI-Small test period: GNN against LightGBM.

Analysis only, no new model and no choice made here. Each laundering test
transaction is given its IBM pattern type from `LI-Small_Patterns.txt` (or
NOT_LISTED). A model "catches" it when its score is at or above its own threshold,
both chosen on validation earlier: LightGBM 0.221 (step 1), GNN 0.698 (Project 4).

Step 1 kept only the 200 alerts' LightGBM scores, so the first run re-fits the
frozen step 1 detector (same data, settings and seeds), checks that it gives the
alerts' scores exactly, and saves the whole test period's scores.

    python -m src.typology_recall
"""

from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import pandas as pd

from src import config

ORDER_REST = ("BIPARTITE", "CYCLE", "RANDOM", "STACK")


def type_by_edge(patterns: Dict, members: Dict[int, set]) -> Dict[int, str]:
    """edge id -> pattern type, from read_patterns and pattern_members output."""
    kind_of_attempt = {attempt: kind for kind, attempt in patterns.values()}
    return {edge: kind_of_attempt[attempt]
            for attempt, edges in members.items() for edge in edges}  # fmt: skip


def laundering_table(edges: pd.DataFrame, types: Dict[int, str]) -> pd.DataFrame:
    """Laundering rows only, with their typology (NOT_LISTED if none)."""
    laundering = edges[edges["is_laundering"] == 1].copy()
    laundering["typology"] = [types.get(int(e), config.UNLISTED)
                              for e in laundering["transaction_id"]]  # fmt: skip
    return laundering


def recall_row(name: str, rows: pd.DataFrame, lightgbm_cut: float,
               gnn_cut: float) -> Dict:  # fmt: skip
    """Counts and recall of each model, and of either, for one typology."""
    lightgbm = rows["lightgbm"] >= lightgbm_cut
    gnn = rows["gnn"] >= gnn_cut
    count = max(len(rows), 1)
    return {"typology": name, "n": len(rows),
            "lightgbm_caught": int(lightgbm.sum()), "gnn_caught": int(gnn.sum()),
            "either_caught": int((lightgbm | gnn).sum()),
            "lightgbm_recall": lightgbm.sum() / count, "gnn_recall": gnn.sum() / count,
            "either_recall": (lightgbm | gnn).sum() / count}  # fmt: skip


def recall_by_typology(laundering: pd.DataFrame, lightgbm_cut: float,
                       gnn_cut: float) -> List[Dict]:  # fmt: skip
    """Focus typologies first, then the rest, NOT_LISTED and all laundering."""
    present = set(laundering["typology"])
    order = [*config.FOCUS_TYPOLOGIES, *ORDER_REST, config.UNLISTED]
    order += sorted(present - set(order))
    rows = [
        recall_row(name, laundering[laundering["typology"] == name], lightgbm_cut,
                   gnn_cut)
        for name in order if name in present
    ]  # fmt: skip
    return rows + [recall_row("ALL", laundering, lightgbm_cut, gnn_cut)]


def markdown(rows: List[Dict]) -> str:
    """The README table."""
    lines = [
        "| Typology (test) | n laundering txns | LightGBM recall (>= 0.221) "
        "| GNN recall (>= 0.698) | Either |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        name = "**%s**" % row["typology"] if row["typology"] in (
            *config.FOCUS_TYPOLOGIES, "ALL") else row["typology"]  # fmt: skip
        cells = [cell(row, model) for model in ("lightgbm", "gnn", "either")]
        lines.append("| %s | %d | %s |" % (name, row["n"], " | ".join(cells)))
    return "\n".join(lines)


def cell(row: Dict, model: str) -> str:
    """ "12 (34%)": caught and recall."""
    return "%d (%.0f%%)" % (row[model + "_caught"], 100 * row[model + "_recall"])


def refit_test_scores(dataset) -> pd.DataFrame:
    """Re-fit the frozen step 1 detector; check it against the alerts; save."""
    from src.features import FEATURE_NAMES, load_or_build
    from src.model import train_detector

    features = load_or_build(dataset.transactions, config.CACHE_DIR, config.RAW_CSV)
    detector = train_detector(features, dataset.truth, dataset.split, FEATURE_NAMES)
    test = dataset.split.test
    scores = pd.DataFrame({
        "transaction_id": dataset.transactions["edge_id"].to_numpy()[test],
        "score": detector.model.predict_proba(features[test])[:, 1]})  # fmt: skip
    check_against_step1(scores, detector.threshold)
    scores.to_parquet(config.ARTIFACTS_DIR / config.LIGHTGBM_TEST_SCORES_FILE)
    return scores


def check_against_step1(scores: pd.DataFrame, threshold: float) -> None:
    """The re-fit must reproduce step 1's threshold and every alert score."""
    summary = json.loads((config.ARTIFACTS_DIR / config.SUMMARY_FILE).read_text())
    alerts = pd.read_parquet(config.ARTIFACTS_DIR / config.ALERTS_FILE)
    joined = alerts.merge(scores, left_on="edge_id", right_on="transaction_id")
    gap = float(np.abs(joined["score_x"] - joined["score_y"]).max())
    if len(joined) != len(alerts) or gap > config.REFIT_SCORE_TOLERANCE or abs(
        threshold - summary["threshold"]) > config.REFIT_SCORE_TOLERANCE:  # fmt: skip
        raise SystemExit("Re-fit differs from step 1 (score gap %g)" % gap)


def main() -> None:
    from src.data import load_dataset
    from src.ground_truth import pattern_members, read_patterns

    dataset = load_dataset(config.RAW_CSV)
    path = config.ARTIFACTS_DIR / config.LIGHTGBM_TEST_SCORES_FILE
    lightgbm = pd.read_parquet(path) if path.exists() else refit_test_scores(dataset)
    gnn = pd.read_parquet(config.ARTIFACTS_DIR / config.GNN_SCORES_FILE)
    gnn = gnn[gnn["period"] == "test"][["transaction_id", "score"]]
    edges = lightgbm.merge(gnn, on="transaction_id", suffixes=("_l", "_g"))
    if len(edges) != len(lightgbm) or len(edges) != len(gnn):
        raise SystemExit("LightGBM and GNN test periods differ")
    edges = edges.rename(columns={"score_l": "lightgbm", "score_g": "gnn"})
    truth = pd.Series(dataset.truth, index=dataset.transactions["edge_id"])
    edges["is_laundering"] = truth.loc[edges["transaction_id"]].to_numpy()
    patterns = read_patterns(config.PATTERNS_FILE)
    types = type_by_edge(patterns, pattern_members(patterns, dataset.transactions))
    summary = json.loads((config.ARTIFACTS_DIR / config.SUMMARY_FILE).read_text())
    rows = recall_by_typology(laundering_table(edges, types), summary["threshold"],
                              config.GNN_HIGH_THRESHOLD)  # fmt: skip
    (config.ARTIFACTS_DIR / config.TYPOLOGY_RECALL_FILE).write_text(
        json.dumps(rows, indent=1, default=float) + "\n")  # fmt: skip
    print(markdown(rows))


if __name__ == "__main__":
    main()
