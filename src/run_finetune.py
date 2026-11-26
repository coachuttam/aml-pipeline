"""V2 step 1: the fine-tuning dataset for the case-note model.

    python -m src.run_finetune pool       # HI-Small detector, top 60,000 (slow)
    python -m src.run_finetune select     # candidates from the pool, with evidence
    python -m src.run_finetune more_false_alarms   # train-only false alarms
    python -m src.run_finetune dev        # 50 LI-Small validation alerts (retrains)
    python -m src.run_finetune generate [--count 20]   # teacher notes (reuses old)
    python -m src.run_finetune rebalance  # train: all laundering, 50% laundering
    python -m src.run_finetune inputs     # dev and test inputs, no note, no truth
    python -m src.run_finetune cues       # cue distributions, train vs dev vs test
    python -m src.run_finetune v3         # train_v3.jsonl: balanced, 35% "unclear"

Output goes to data/finetune/ (git-ignored): train.jsonl, validation.jsonl (chat
format), meta.jsonl (truth per example, for analysis only), report.json,
cue_report.json, dev_inputs.jsonl and test_inputs.jsonl (inputs only). A batch with
--count goes to data/finetune/sample/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src import config
from src.finetune_alerts import add_false_alarms, build_pool, select_candidates
from src.finetune_cues import build_cue_report
from src.finetune_dataset import (
    example_for,
    generate_records,
    student_messages,
    token_count,
)
from src.finetune_rebalance import rebalance_files
from src.finetune_report import build_report
from src.finetune_reuse import load_old_notes
from src.finetune_select import pick_sample
from src.finetune_v3 import build_train_v3, example_tokens, l4_hours
from src.llm_client import make_client
from src.run_dev import build_dev_set

STEPS = [
    "pool",
    "select",
    "more_false_alarms",
    "dev",
    "generate",
    "rebalance",
    "inputs",
    "cues",
    "v3",
]
V3_EPOCHS = 2  # the notebook's EPOCHS for the typology2 run


def write_jsonl(path: Path, items: List[Dict]) -> None:
    """One JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))


def write_dataset(folder: Path, candidates: Path, records: List[Dict]) -> Dict:
    """Chat files per split, the truth beside them, and the report."""
    kept = [r for r in records if not r["rejected"]]
    counts: Dict[str, List[int]] = {}
    for split in ("train", "validation"):
        mine = [r for r in kept if r["split"] == split]
        examples = [example_for(candidates, r) for r in mine]
        counts[split] = [token_count(e) for e in examples]
        write_jsonl(folder / ("%s.jsonl" % split), examples)
    meta = [{k: v for k, v in r.items() if k != "note"} for r in kept]
    write_jsonl(folder / "meta.jsonl", meta)
    report = build_report(records, counts)
    (folder / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def write_dev_inputs(dev_folder: Path, output: Path) -> int:
    """The alerts of a folder as system + user messages only (no note, no truth)."""
    alerts = pd.read_parquet(dev_folder / config.ALERTS_FILE)
    lines = []
    for alert in alerts.to_dict("records"):
        name = "%s.json" % alert["alert_id"]
        evidence = json.loads((dev_folder / config.EVIDENCE_FOLDER / name).read_text())
        lines.append(
            {
                "alert_id": alert["alert_id"],
                "messages": student_messages(alert, evidence),
            }
        )
    write_jsonl(output, lines)
    return len(lines)


def generate(count: Optional[int]) -> Dict:
    """Teacher notes for `count` candidates (all when None), then the files."""
    data = config.FINETUNE_DIR
    candidates = data / config.FINETUNE_CANDIDATES_FOLDER
    chosen = pd.read_parquet(candidates / config.GROUND_TRUTH_FILE)
    if count:
        scores = pd.read_parquet(candidates / config.ALERTS_FILE)[["alert_id", "score"]]
        chosen = pick_sample(chosen.merge(scores, on="alert_id"), count)
    old_notes = load_old_notes(data / "old_v2", data / "candidates_v2")
    records = generate_records(make_client(), candidates, chosen, old_notes)
    folder = data / ("sample" if count else "")
    return write_dataset(folder, candidates, records)


def build_v3(data: Path, candidates: Path) -> Dict:
    """train_v3.jsonl, the mix before and after, and the L4 time estimate."""
    from src.finetune_diagnose import load_token_counter, target_mix

    truth = pd.read_parquet(candidates / config.GROUND_TRUTH_FILE)
    report = {
        "after": build_train_v3(data, dict(zip(truth["alert_id"], truth["attempt_id"])))
    }
    meta = [json.loads(line) for line in (data / "meta.jsonl").open()]
    names = ("train.jsonl", config.FINETUNE_V3_TRAIN, "validation.jsonl")
    files = {n: [json.loads(x) for x in (data / n).open()] for n in names}
    report["before"] = target_mix({"train": files["train.jsonl"]}, meta)["train"]
    counter = load_token_counter()
    tokens = {n: sum(example_tokens(e, counter["count"]) for e in lines)
              for n, lines in files.items()}  # fmt: skip
    report["tokens"] = {"tokenizer": counter["name"], **tokens}
    report["l4_hours_estimate"] = l4_hours(
        tokens[config.FINETUNE_V3_TRAIN], tokens["validation.jsonl"], V3_EPOCHS
    )
    (data / "train_v3_report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("--count", type=int, default=None)
    arguments = parser.parse_args()
    data = config.FINETUNE_DIR
    dev_folder = config.ARTIFACTS_DIR / config.FINETUNE_DEV_FOLDER
    candidates = data / config.FINETUNE_CANDIDATES_FOLDER
    if arguments.step == "pool":
        build_pool(candidates / "pool")
    elif arguments.step == "select":
        select_candidates(candidates / "pool", candidates, dev_folder)
    elif arguments.step == "more_false_alarms":
        count = arguments.count or config.FINETUNE_EXTRA_FALSE_ALARMS
        print("added:", add_false_alarms(candidates / "pool", candidates, count))
    elif arguments.step == "dev":
        build_dev_set(
            config.ARTIFACTS_DIR, config.RAW_CSV, config.FINETUNE_DEV_ALERTS,
            config.FINETUNE_DEV_FOLDER,
        )  # fmt: skip
    elif arguments.step == "rebalance":
        print(json.dumps(rebalance_files(data), indent=1))
    elif arguments.step == "inputs":
        print("dev inputs:", write_dev_inputs(dev_folder, data / "dev_inputs.jsonl"))
        test_file = data / config.FINETUNE_TEST_INPUTS
        print("test inputs:", write_dev_inputs(config.ARTIFACTS_DIR, test_file))
    elif arguments.step == "cues":
        report = build_cue_report()
        (data / "cue_report.json").write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps(report, indent=1))
    elif arguments.step == "v3":
        print(json.dumps(build_v3(data, candidates), indent=1))
    else:
        print(json.dumps(generate(arguments.count), indent=1))


if __name__ == "__main__":
    main()
