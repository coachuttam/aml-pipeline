"""V2 note writer: the frozen typology2 student writes the case notes the UI shows.

Three steps, the middle one on a GPU (Colab L4 or similar), the others on a CPU:

    python -m src.note_writer inputs   --out data/finetune/new_inputs.jsonl
    python -m src.note_writer generate --inputs data/finetune/new_inputs.jsonl \
        --out data/finetune/new_raw.jsonl          # GPU; adapter from config
    python -m src.note_writer notes    --raw data/finetune/new_raw.jsonl

Offline (no GPU, no internet): the same student, merged and quantised to a Q4_K_M
GGUF, served by a local Ollama (ollama/Modelfile, docs/OFFLINE_DEMO.md):

    python -m src.note_writer ollama --alert A0001   # one alert, end to end, timed

The answer goes through the same parsing and fact check (student_record). Ollama
answers are demo output in data/finetune/ollama_demo_raw.jsonl and are never scored.

`notes` parses and fact-checks each answer against its own evidence and writes
artifacts/notes_student.json, the file the UI reads when NOTE_WRITER is "student"
(the default; "prompt_v3" switches back to V1). It reads no ground truth. The
student's typology is a suggestion: right on 18 of 37 visible test patterns, so the
analyst confirms or corrects it (typology_text).

Example:
    typology_text({"typology": "FAN_IN"})   # "Suggested typology: FAN_IN ..."
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src import config
from src.score_student import student_record

STUDENT_SCRIPT = Path(__file__).with_name("student_generate.py")


def typology_text(note: Dict, writer: Optional[str] = None) -> str:
    """How the UI states the typology: a suggestion the analyst confirms."""
    writer = writer or config.NOTE_WRITER
    if note["typology"] == config.UNCLEAR:
        return "Suggested typology: unclear (the evidence does not show a pattern)"
    who = "fine-tuned student" if writer == "student" else "LLM hint"
    return "Suggested typology: %s (%s; analyst confirms)" % (note["typology"], who)


def write_notes(folder: Path, raw_path: Path) -> Dict:
    """Fact-checked student notes for every alert of a folder (no ground truth)."""
    alerts = pd.read_parquet(folder / config.ALERTS_FILE)
    answers = {}
    for line in raw_path.read_text().splitlines():
        answer = json.loads(line)
        answers[answer["alert_id"]] = answer
    missing = {"text": "", "input_tokens": 0, "output_tokens": 0}
    records = []
    for alert in alerts.to_dict("records"):
        name = "%s.json" % alert["alert_id"]
        evidence = json.loads((folder / config.EVIDENCE_FOLDER / name).read_text())
        answer = answers.get(alert["alert_id"], missing)
        records.append(student_record(alert, evidence, answer))
    (folder / config.STUDENT_NOTES_FILE).write_text(json.dumps(records, indent=1))
    needing_review = sum(r["needs_review"] for r in records)
    return {"alerts": len(records), "needs_review": needing_review}


def ollama_record(alert: Dict, evidence: Dict) -> Dict:
    """One alert's note from the local Ollama model, fact-checked like any other."""
    from src.finetune_dataset import student_messages
    from src.ollama_generate import write_note

    item = {
        "alert_id": alert["alert_id"],
        "messages": student_messages(alert, evidence),
    }
    line = write_note(item, config.OLLAMA_MODEL, config.OLLAMA_HOST_URL)
    record = student_record(alert, evidence, line)
    record["seconds"] = line["seconds"]
    record["note_version"] = config.OLLAMA_NOTE_VERSION
    return record


def run_ollama_alert(folder: Path, alert_id: str) -> Dict:
    """`ollama --alert`: write, check and time one note; keep the raw answer."""
    alerts = pd.read_parquet(folder / config.ALERTS_FILE)
    alert = alerts[alerts["alert_id"] == alert_id].iloc[0].to_dict()
    name = "%s.json" % alert_id
    evidence = json.loads((folder / config.EVIDENCE_FOLDER / name).read_text())
    record = ollama_record(alert, evidence)
    raw = config.FINETUNE_DIR / config.OLLAMA_DEMO_RAW
    raw.parent.mkdir(parents=True, exist_ok=True)
    with raw.open("a") as handle:
        handle.write(json.dumps({"alert_id": alert_id, "note": record["note"],
                                 "seconds": record["seconds"]}) + "\n")  # fmt: skip
    return record


def print_ollama_result(record: Dict) -> None:
    """The record, then one summary line: time, tokens, fact check."""
    print(json.dumps(record, indent=1))
    check = record["violations"] if record["needs_review"] else "clean"
    values = (record["alert_id"], record["seconds"], record["output_tokens"], check)
    print("%s: %.0f s, %d output tokens, fact check: %s" % values)


def generate_command(inputs: Path, output: Path) -> list:
    """The GPU command: student_generate.py with the frozen adapter."""
    adapter = str(config.STUDENT_ADAPTER_DIR)
    return [sys.executable, str(STUDENT_SCRIPT), "--inputs", str(inputs),
            "--output", str(output), "--adapter", adapter]  # fmt: skip


def main() -> None:
    from src.run_finetune import write_dev_inputs

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["inputs", "generate", "notes", "ollama"])
    parser.add_argument("--alert", help="ollama: the one alert to write a note for")
    parser.add_argument("--folder", type=Path, default=config.ARTIFACTS_DIR)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args()
    if arguments.step == "inputs":
        print("%d inputs" % write_dev_inputs(arguments.folder, arguments.out))
    elif arguments.step == "generate":
        if not config.STUDENT_ADAPTER_DIR.exists():
            raise SystemExit("No adapter at %s" % config.STUDENT_ADAPTER_DIR)
        subprocess.run(generate_command(arguments.inputs, arguments.out), check=True)
    elif arguments.step == "ollama":
        print_ollama_result(run_ollama_alert(arguments.folder, arguments.alert))
    else:
        print(write_notes(arguments.folder, arguments.raw))


if __name__ == "__main__":
    main()
