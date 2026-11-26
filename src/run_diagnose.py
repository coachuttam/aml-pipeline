"""V2 diagnosis commands (CPU only, no retraining, never the test answers).

    python -m src.run_diagnose val       # HI-Small validation set + val_inputs.jsonl
    python -m src.run_diagnose format    # inputs of every file vs train's format
    python -m src.run_diagnose targets   # target mix and typology share of loss tokens
    python -m src.run_diagnose rules     # rule baseline, tuned on dev, also run on val

Reports go to data/finetune/diagnosis.json (one key per command).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src import config
from src.finetune_diagnose import (
    format_report,
    load_token_counter,
    target_mix,
    typology_token_share,
)
from src.finetune_val import read_meta, write_val_folder
from src.run_finetune import write_dev_inputs
from src.typology_rules import RuleSettings, detect_typology, score_rules, tune

DATA = config.FINETUNE_DIR
VAL_FOLDER = DATA / config.FINETUNE_VAL_FOLDER
DEV_FOLDER = config.ARTIFACTS_DIR / config.FINETUNE_DEV_FOLDER


def read_lines(path: Path) -> List[Dict]:
    """The JSON lines of a file."""
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def build_val() -> Dict:
    """The val folder and its inputs; checks they equal validation.jsonl's prompts."""
    count = write_val_folder(
        DATA / config.FINETUNE_CANDIDATES_FOLDER,
        read_meta(DATA / "meta.jsonl"),
        VAL_FOLDER,
    )
    write_dev_inputs(VAL_FOLDER, DATA / config.FINETUNE_VAL_INPUTS)
    trained = {
        e["alert_id"]: e["messages"][:2] for e in read_lines(DATA / "validation.jsonl")
    }
    inputs = read_lines(DATA / config.FINETUNE_VAL_INPUTS)
    same = sum(trained[i["alert_id"]] == i["messages"] for i in inputs)
    truth = pd.read_parquet(VAL_FOLDER / config.GROUND_TRUTH_FILE).merge(
        pd.read_parquet(VAL_FOLDER / config.PATTERN_VISIBLE_FILE), on="alert_id"
    )
    visible = truth[(truth["is_laundering"] == 1) & truth["pattern_visible"]]
    return {
        "alerts": count,
        "inputs_equal_to_validation_jsonl": same,
        "visible_typologies": visible["pattern_type"].value_counts().to_dict(),
        "false_alarms": int((truth["is_laundering"] == 0).sum()),
    }


def check_format() -> Dict:
    """Format of dev, test and val inputs against train."""
    names = {"train": "train.jsonl", "validation": "validation.jsonl",
             "dev": "dev_inputs.jsonl", "test": config.FINETUNE_TEST_INPUTS,
             "val": config.FINETUNE_VAL_INPUTS}  # fmt: skip
    files = {k: read_lines(DATA / v) for k, v in names.items() if (DATA / v).exists()}
    return format_report(files)


def check_targets() -> Dict:
    """Target mix of train and validation, and the typology share of the loss."""
    train = read_lines(DATA / "train.jsonl")
    examples = {"train": train, "validation": read_lines(DATA / "validation.jsonl")}
    counter = load_token_counter()
    return {
        "mix": target_mix(examples, read_meta(DATA / "meta.jsonl")),
        "tokenizer": counter["name"],
        "train_loss_tokens": typology_token_share(train, counter["count"]),
    }


def folder_case(folder: Path) -> Dict:
    """Evidence by alert id and truth rows (with pattern_visible) of one folder."""
    truth = pd.read_parquet(folder / config.GROUND_TRUTH_FILE).merge(
        pd.read_parquet(folder / config.PATTERN_VISIBLE_FILE), on="alert_id"
    )
    evidence = {
        alert_id: json.loads(
            (folder / config.EVIDENCE_FOLDER / (alert_id + ".json")).read_text()
        )
        for alert_id in truth["alert_id"]
    }
    return {"evidence": evidence, "truth": truth.to_dict("records")}


def run_rules() -> Dict:
    """Tune on dev, then the same settings on val; constant baselines beside them."""
    grid = [
        RuleSettings(window, degree, payers)
        for window in config.RULE_WINDOW_HOURS
        for degree in config.RULE_MIN_DEGREES
        for payers in config.RULE_HUB_MIN_PAYERS
    ]
    dev = folder_case(DEV_FOLDER)
    settings, dev_result = tune(dev["evidence"], dev["truth"], grid)
    strict = [g for g in grid if g.hub_min_payers >= 2]  # a real gather, not one payer
    strict_settings, strict_result = tune(dev["evidence"], dev["truth"], strict)
    report = {
        "settings": vars(settings),
        "dev": dev_result,
        "strict_hub": {"settings": vars(strict_settings), "dev": strict_result},
    }
    cases = {"dev": dev}
    if VAL_FOLDER.exists():
        cases["val"] = folder_case(VAL_FOLDER)
    for name, case in cases.items():
        predicted = {
            i: detect_typology(e, settings) for i, e in case["evidence"].items()
        }
        report[name] = score_rules(predicted, case["truth"])
        report[name]["predicted"] = dict(pd.Series(predicted).value_counts())
        constant = {i: "GATHER_SCATTER" for i in case["evidence"]}
        report[name]["always_gather_scatter"] = score_rules(constant, case["truth"])
    return report


COMMANDS = {"val": build_val, "format": check_format, "targets": check_targets,
            "rules": run_rules}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=list(COMMANDS))
    command = parser.parse_args().command
    result = COMMANDS[command]()
    path = DATA / "diagnosis.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    saved[command] = result
    path.write_text(json.dumps(saved, indent=1, default=int) + "\n")
    print(json.dumps(result, indent=1, default=int))


if __name__ == "__main__":
    main()
