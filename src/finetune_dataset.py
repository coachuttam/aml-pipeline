"""Turn alerts, evidence and teacher notes into chat-format fine-tuning examples.

One example is {"messages": [system, user, assistant]}. The system and user messages
are exactly what the student gets at test time (the v3 prompt and the alert input);
the assistant message is the teacher's case note as JSON. Nothing about the truth
is in the system or user message.

Example:
    record = make_record(client, alert, evidence, truth_row)
    example = chat_example(student_messages(alert, evidence), record["note"])
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src import config
from src.case_note_input import build_user_message, load_prompt, response_format
from src.llm_client import request_note
from src.teacher import (
    build_teacher_message,
    is_pattern_visible,
    parse_teacher_note,
    privileged_truth,
    rejection_reasons,
)


def compact_json(text: str) -> str:
    """The same JSON without indentation (Qwen needs a quarter fewer tokens)."""
    return json.dumps(json.loads(text), separators=(",", ":"), ensure_ascii=False)


def student_messages(alert: Dict, evidence: Dict) -> List[Dict]:
    """System and user message of the student: no label, no pattern, no truth.

    The user message holds exactly the content V1 and the teacher get, only written
    without indentation: with Qwen's one-token-per-digit numbers the indented form
    is 8.8k tokens on average, over the 8,192 limit for 96% of the examples.
    """
    return [
        {"role": "system", "content": load_prompt(config.STUDENT_PROMPT_VERSION)},
        {"role": "user", "content": compact_json(build_user_message(alert, evidence))},
    ]


def note_as_text(note: Dict) -> str:
    """The note as the student must write it: schema fields only, in schema order."""
    fields = response_format()["json_schema"]["schema"]["properties"]
    return json.dumps({name: note[name] for name in fields}, ensure_ascii=False)


def chat_example(messages: List[Dict], note: Dict) -> Dict:
    """One fine-tuning line: system, user and the teacher's note as assistant."""
    return {
        "messages": messages + [{"role": "assistant", "content": note_as_text(note)}]
    }


def make_record(client: Any, alert: Dict, evidence: Dict, row: Dict) -> Dict:
    """Ask the teacher for one alert; keep the note and why it was rejected, if so."""
    truth = privileged_truth(row, evidence)
    student = build_user_message(alert, evidence)
    result = request_note(
        client, build_teacher_message(student, truth), config.TEACHER_PROMPT_VERSION
    )
    note = parse_teacher_note(result["content"])
    return {
        "alert_id": alert["alert_id"],
        "split": row["split"],
        "is_laundering": int(row["is_laundering"]),
        "pattern_type": row["pattern_type"],
        "typology_to_use": truth["typology_to_use"],
        "pattern_visible": is_pattern_visible(truth),
        "note": note,
        "rejected": rejection_reasons(note, evidence, truth),
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
    }


def load_evidence(folder: Path, alert_id: str) -> Dict:
    """One evidence file of a candidate folder."""
    path = folder / config.EVIDENCE_FOLDER / ("%s.json" % alert_id)
    return json.loads(path.read_text())


def generate_records(client: Any, folder: Path, chosen: pd.DataFrame) -> List[Dict]:
    """Teacher records for the chosen alerts, several API calls at once."""
    alerts = pd.read_parquet(folder / config.ALERTS_FILE).set_index("alert_id")
    truth = chosen.set_index("alert_id")

    def one(alert_id: str) -> Dict:
        alert = {**alerts.loc[alert_id].to_dict(), "alert_id": alert_id}
        row = truth.loc[alert_id].to_dict()
        return make_record(client, alert, load_evidence(folder, alert_id), row)

    with ThreadPoolExecutor(config.TEACHER_WORKERS) as pool:
        return list(pool.map(one, list(chosen["alert_id"])))


def example_for(folder: Path, record: Dict) -> Dict:
    """The chat example of a kept record (re-reads its alert and evidence)."""
    alerts = pd.read_parquet(folder / config.ALERTS_FILE).set_index("alert_id")
    alert = {**alerts.loc[record["alert_id"]].to_dict(), "alert_id": record["alert_id"]}
    evidence = load_evidence(folder, record["alert_id"])
    return chat_example(student_messages(alert, evidence), record["note"])


def token_count(example: Dict) -> int:
    """Tokens of all the message texts (tiktoken; a few per message are not counted)."""
    import tiktoken

    encoding = tiktoken.get_encoding(config.TOKEN_ENCODING)
    return sum(len(encoding.encode(m["content"])) for m in example["messages"])


def length_summary(counts: List[int]) -> Dict:
    """Median, 95th percentile and maximum of the token counts."""
    if not counts:
        return {}
    return {
        "p50": int(np.percentile(counts, 50)),
        "p95": int(np.percentile(counts, 95)),
        "max": int(max(counts)),
    }


def teacher_cost(records: List[Dict]) -> float:
    """Dollars for the teacher calls, from the config price list."""
    paid_in = sum(r["input_tokens"] for r in records)
    paid_out = sum(r["output_tokens"] for r in records)
    return (
        paid_in * config.LLM_PRICE_INPUT_PER_MILLION
        + paid_out * config.LLM_PRICE_OUTPUT_PER_MILLION
    ) / 1e6
