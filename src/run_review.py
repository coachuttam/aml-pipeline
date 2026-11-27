"""Review of a student run and evidence v2 (CPU only, no retraining, no test answers).

    python -m src.run_review confusion [--run typology2]   # val and dev tables
    python -m src.run_review invented                      # dev invented facts
    python -m src.run_review evidence2                     # v2 evidence + inputs

`confusion` reads data/finetune/{val,dev}_raw_Qwen2.5-7B-Instruct_<run>.jsonl.
`invented` reads artifacts/dev_finetune/notes_student.json (score_student writes it).
`evidence2` writes evidence_v2/ beside evidence/ for dev (artifacts/dev_finetune/) and
test (artifacts/), then data/finetune/dev_inputs_v2.jsonl and test_inputs_v2.jsonl, and
compares v2 with v1. It reads test ground truth for coverage only; rule settings are
tuned on dev. Reports go to data/finetune/review.json (one key per command).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src import config
from src.evidence import EvidenceIndex, build_evidence, write_evidence
from src.evidence_v2 import coverage_summary, hop_mix, rule_scores, visible_flags
from src.student_review import (
    confusion_table,
    invented_facts,
    table_markdown,
    unchecked_amounts,
)
from src.teacher import parse_teacher_note
from src.typology_rules import RuleSettings, tune

DATA = config.FINETUNE_DIR
FOLDERS = {
    "val": DATA / config.FINETUNE_VAL_FOLDER,
    "dev": config.ARTIFACTS_DIR / config.FINETUNE_DEV_FOLDER,
    "test": config.ARTIFACTS_DIR,
}
MODEL = "Qwen2.5-7B-Instruct"


def truth_rows(folder: Path) -> List[Dict]:
    """Ground truth with pattern_visible (v1 evidence) of one alert folder."""
    truth = pd.read_parquet(folder / config.GROUND_TRUTH_FILE)
    flags = pd.read_parquet(folder / config.PATTERN_VISIBLE_FILE)
    return truth.merge(flags, on="alert_id").to_dict("records")


def predicted_typologies(raw_path: Path) -> Dict[str, str]:
    """alert_id -> typology of each parsed answer (an invalid answer is left out)."""
    predicted = {}
    for line in raw_path.read_text().splitlines():
        raw = json.loads(line)
        note = None if raw.get("truncated") else parse_teacher_note(raw["text"])
        if note is not None:
            predicted[raw["alert_id"]] = note["typology"]
    return predicted


def run_confusion(run: str) -> Dict:
    """Confusion table (true vs predicted typology) for val and dev."""
    report = {}
    for name in ("val", "dev"):
        raw = DATA / ("%s_raw_%s_%s.jsonl" % (name, MODEL, run))
        table = confusion_table(predicted_typologies(raw), truth_rows(FOLDERS[name]))
        report[name] = {"table": table, "markdown": table_markdown(table)}
        print("\n%s (%s)\n%s" % (name, raw.name, report[name]["markdown"]))
    return report


def read_evidence(folder: Path) -> Dict[str, Dict]:
    """alert_id -> evidence document of one evidence folder."""
    return {p.stem: json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))}


def run_invented() -> Dict:
    """Every dev violation with its kind and cause, and amounts the check skips."""
    folder = FOLDERS["dev"]
    records = json.loads((folder / "notes_student.json").read_text())
    evidence = read_evidence(folder / config.EVIDENCE_FOLDER)
    rows = invented_facts(records, evidence)
    skipped = {
        r["alert_id"]: unchecked_amounts(r["note"], evidence[r["alert_id"]])
        for r in records
        if r["note"]
    }
    report = {
        "total": len(rows),
        "by_kind": pd.Series([r["kind"] for r in rows]).value_counts().to_dict(),
        "by_cause": pd.Series([r["cause"] for r in rows]).value_counts().to_dict(),
        "rows": rows,
        "unchecked_one_decimal_amounts": {k: v for k, v in skipped.items() if v},
    }
    print(json.dumps(report, indent=1))
    return report


def build_v2_folder(index: EvidenceIndex, folder: Path) -> List[Dict]:
    """evidence_v2/ for every alert of a folder; the documents."""
    alerts = pd.read_parquet(folder / config.ALERTS_FILE)
    documents = []
    for alert_id, row in zip(alerts["alert_id"], alerts["row"]):
        document = build_evidence(
            index, int(row), alert_id, config.EVIDENCE_V2_STRATEGY
        )
        write_evidence(document, folder / config.EVIDENCE_V2_FOLDER)
        documents.append(document)
    return documents


def rule_settings_from_dev(dev_documents: List[Dict], truth: List[Dict]) -> Dict:
    """Best rule settings on one dev evidence version (dev only, never test)."""
    grid = [
        RuleSettings(window, degree, payers)
        for window in config.RULE_WINDOW_HOURS
        for degree in config.RULE_MIN_DEGREES
        for payers in config.RULE_HUB_MIN_PAYERS
    ]
    evidence = {d["alert_id"]: d for d in dev_documents}
    return tune(evidence, truth, grid)[0]


def compare_versions(documents: Dict, members: Dict) -> Dict:
    """Hop mix, coverage and rules for v1 and v2 of dev and test."""
    report: Dict = {}
    truth = {n: pd.read_parquet(FOLDERS[n] / config.GROUND_TRUTH_FILE)
             for n in ("dev", "test")}  # fmt: skip
    flagged = {(n, v): visible_flags(documents[n][v], truth[n], members)
               for n in truth for v in ("v1", "v2")}  # fmt: skip
    settings = {v: rule_settings_from_dev(documents["dev"][v], flagged[("dev", v)])
                for v in ("v1", "v2")}  # fmt: skip
    report["rule_settings_tuned_on_dev"] = {v: vars(s) for v, s in settings.items()}
    for name in truth:
        for version in ("v1", "v2"):
            docs = documents[name][version]
            report["%s_%s" % (name, version)] = {
                "hops": hop_mix(docs),
                "coverage": coverage_summary(docs, truth[name], members),
                **rule_scores(docs, flagged[(name, version)], settings[version]),
            }
    return report


def run_evidence2() -> Dict:
    """Build v2 evidence and inputs for dev and test, then compare with v1."""
    from src.data import load_dataset
    from src.ground_truth import pattern_members, read_patterns
    from src.run_finetune import write_dev_inputs

    dataset = load_dataset(config.RAW_CSV)
    index = EvidenceIndex(dataset.transactions)
    members = pattern_members(read_patterns(config.PATTERNS_FILE), dataset.transactions)
    documents = {}
    for name in ("dev", "test"):
        folder = FOLDERS[name]
        v1 = list(read_evidence(folder / config.EVIDENCE_FOLDER).values())
        documents[name] = {"v1": v1, "v2": build_v2_folder(index, folder)}
        output = DATA / ("%s_inputs_v2.jsonl" % name)
        write_dev_inputs(folder, output, config.EVIDENCE_V2_FOLDER)
    report = compare_versions(documents, members)
    print(json.dumps(report, indent=1, default=int))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["confusion", "invented", "evidence2"])
    parser.add_argument("--run", default="typology2")
    arguments = parser.parse_args()
    if arguments.command == "confusion":
        result = run_confusion(arguments.run)
    elif arguments.command == "invented":
        result = run_invented()
    else:
        result = run_evidence2()
    path = DATA / "review.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    saved[arguments.command] = result
    path.write_text(json.dumps(saved, indent=1, default=int) + "\n")


if __name__ == "__main__":
    main()
