"""Quantisation check, dev only: the Q4_K_M GGUF (Ollama) against the LoRA adapter.

Both answer the same 50 dev alerts (LI-Small validation period); test is never used.
Each answer goes through the same parsing and fact check (score_student's
student_record). Reported for each model: valid JSON, invented facts per checked
claim, typology right on the 19 dev alerts with a visible pattern (dev ground truth,
evaluation only); and between them: same typology on alerts valid in both, and
identical answers. The pass rules (config.QUANT_*) were fixed before the GGUF ran.

    python -m src.quant_check --gguf-raw data/finetune/dev_raw_<model>_q4km.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src import config
from src.score_student import student_record
from src.student_review import true_class

DEV_FOLDER = config.ARTIFACTS_DIR / config.FINETUNE_DEV_FOLDER
ADAPTER_RAW = config.FINETUNE_DIR / "dev_raw_Qwen2.5-7B-Instruct_typology2.jsonl"
REPORT_FILE = config.FINETUNE_DIR / "quant_check.json"


def refuse_non_dev(path: Path) -> None:
    """Only dev answer files: test stays the adapter's single scored run."""
    if not path.name.startswith("dev_"):
        raise SystemExit("The quantisation check runs on dev answers only: %s" % path)


def read_raw(path: Path) -> Dict[str, Dict]:
    """alert_id -> answer line (a later line, e.g. a retry, replaces an earlier one)."""
    lines = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {line["alert_id"]: line for line in lines}


def scored(raw: Dict[str, Dict], folder: Path) -> Dict[str, Dict]:
    """alert_id -> fact-checked record, for every alert of the folder."""
    alerts = pd.read_parquet(folder / config.ALERTS_FILE)
    missing = {"text": "", "input_tokens": 0, "output_tokens": 0}
    records = {}
    for alert in alerts.to_dict("records"):
        name = "%s.json" % alert["alert_id"]
        evidence = json.loads((folder / config.EVIDENCE_FOLDER / name).read_text())
        line = raw.get(alert["alert_id"], missing)
        records[alert["alert_id"]] = student_record(alert, evidence, line)
    return records


def summary(records: Dict[str, Dict], truth: List[Dict]) -> Dict:
    """Valid answers, invented facts per claim, typology right on visible patterns."""
    notes = {k: r["note"] for k, r in records.items() if r["note"]}
    invented = sum(len(r["violations"]) for r in records.values())
    claims = sum(r["claims"] for r in records.values())
    visible = [row for row in truth if row["is_laundering"] and row["pattern_visible"]]
    right = sum(notes.get(row["alert_id"], {}).get("typology") == true_class(row)
                for row in visible)  # fmt: skip
    return {"alerts": len(records), "valid": len(notes), "invented": invented,
            "claims": claims, "invented_rate": invented / max(claims, 1),
            "typology_right_visible": right, "visible": len(visible)}  # fmt: skip


def agreement(first: Dict[str, Dict], second: Dict[str, Dict]) -> Dict:
    """Same typology on alerts valid in both; identical notes."""
    both = [k for k in first if first[k]["note"] and second.get(k, {}).get("note")]
    same = sum(first[k]["note"]["typology"] == second[k]["note"]["typology"]
               for k in both)  # fmt: skip
    identical = sum(first[k]["note"] == second[k]["note"] for k in both)
    return {"both_valid": len(both), "same_typology": same,
            "typology_agreement": same / max(len(both), 1),
            "identical_notes": identical}  # fmt: skip


def verdict(adapter: Dict, gguf: Dict, agree: Dict) -> Dict[str, bool]:
    """The three pass rules, fixed in config before the GGUF answers were seen."""
    return {
        "json_validity": gguf["valid"] >= adapter["valid"] - config.QUANT_MAX_JSON_DROP,
        "invented_facts": gguf["invented_rate"]
        <= config.QUANT_MAX_INVENTED_RATIO * adapter["invented_rate"],
        "typology_agreement": agree["typology_agreement"]
        >= config.QUANT_MIN_TYPOLOGY_AGREEMENT,
    }


def compare(adapter_raw: Path, gguf_raw: Path, folder: Path, truth: List[Dict]) -> Dict:
    """The whole report for two dev answer files."""
    for path in (adapter_raw, gguf_raw):
        refuse_non_dev(path)
    first = scored(read_raw(adapter_raw), folder)
    second = scored(read_raw(gguf_raw), folder)
    report = {"adapter": summary(first, truth), "gguf_q4km": summary(second, truth),
              "agreement": agreement(first, second)}  # fmt: skip
    report["passed"] = verdict(report["adapter"], report["gguf_q4km"],
                               report["agreement"])  # fmt: skip
    return report


def main() -> None:
    from src.run_review import truth_rows

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gguf-raw", type=Path, required=True)
    parser.add_argument("--adapter-raw", type=Path, default=ADAPTER_RAW)
    arguments = parser.parse_args()
    report = compare(arguments.adapter_raw, arguments.gguf_raw, DEV_FOLDER,
                     truth_rows(DEV_FOLDER))  # fmt: skip
    REPORT_FILE.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
