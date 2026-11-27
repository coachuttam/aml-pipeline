"""Write student case notes with the quantised GGUF model through a local Ollama.

Self-contained on purpose (standard library only, no imports from this repo) so it can
run in Colab next to the notebook as well as on the Mac:

    python ollama_generate.py --inputs dev_inputs.jsonl --output dev_raw_q4km.jsonl

Same input and output lines as student_generate.py ({"alert_id", "messages"} in,
{"alert_id", "text", "input_tokens", "output_tokens", "truncated", "retried"} out,
plus "seconds"), the same decoding (greedy, at most 1,400 new tokens: both set in
ollama/Modelfile) and the same rule: an invalid answer is retried once with
repetition penalty 1.1. Finished alerts are skipped, so a rerun resumes.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

MODEL = "aml-student-typology2"  # the name given to `ollama create`
HOST = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")
RETRY_REPETITION_PENALTY = (
    1.1  # as student_generate.py, used once for an invalid answer
)
REQUEST_TIMEOUT_SECONDS = 1800  # a slow laptop can take minutes per note
NOTE_FIELDS = {"typology", "key_accounts", "cited_txn_ids", "red_flags",
               "investigator_questions", "reason"}  # fmt: skip


def request_body(messages: List[Dict], model: str, penalty: Optional[float]) -> Dict:
    """An /api/chat request; decoding settings come from the Modelfile."""
    body: Dict = {"model": model, "messages": messages, "stream": False}
    if penalty is not None:
        body["options"] = {"repeat_penalty": penalty}
    return body


def chat(messages: List[Dict], model: str = MODEL, host: str = HOST,
         penalty: Optional[float] = None) -> Dict:  # fmt: skip
    """One non-streamed answer from Ollama's chat API."""
    data = json.dumps(request_body(messages, model, penalty)).encode("utf-8")
    request = urllib.request.Request(
        host.rstrip("/") + "/api/chat", data=data,
        headers={"Content-Type": "application/json"},
    )  # fmt: skip
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as reply:
        return json.loads(reply.read().decode("utf-8"))


def answer_line(alert_id: str, reply: Dict, seconds: float, retried: bool) -> Dict:
    """Ollama's reply in student_generate.py's line format."""
    return {
        "alert_id": alert_id,
        "text": reply["message"]["content"],
        "input_tokens": int(reply.get("prompt_eval_count", 0)),
        "output_tokens": int(reply.get("eval_count", 0)),
        "truncated": reply.get("done_reason") == "length",
        "retried": retried,
        "seconds": round(seconds, 1),
    }


def is_valid(line: Dict) -> bool:
    """An answer that stopped by itself and is JSON with exactly the six fields."""
    if line.get("truncated"):
        return False
    try:
        note = json.loads(line["text"])
    except ValueError:
        return False
    return isinstance(note, dict) and set(note) == NOTE_FIELDS


def write_note(item: Dict, model: str = MODEL, host: str = HOST) -> Dict:
    """One alert's answer, retried once (penalty 1.1) if it is invalid."""
    start = time.time()
    line = answer_line(item["alert_id"], chat(item["messages"], model, host),
                       time.time() - start, False)  # fmt: skip
    if is_valid(line):
        return line
    start = time.time()
    reply = chat(item["messages"], model, host, RETRY_REPETITION_PENALTY)
    return answer_line(item["alert_id"], reply, time.time() - start, True)


def load_lines(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--host", default=HOST)
    arguments = parser.parse_args()
    done = {line["alert_id"] for line in load_lines(arguments.output)}
    todo = [i for i in load_lines(arguments.inputs) if i["alert_id"] not in done]
    with arguments.output.open("a") as handle:
        for number, item in enumerate(todo, 1):
            line = write_note(item, arguments.model, arguments.host)
            handle.write(json.dumps(line) + "\n")
            handle.flush()
            print("%d / %d  %s  %.0f s%s" % (number, len(todo), item["alert_id"],
                  line["seconds"], "  (retried)" if line["retried"] else ""),
                  flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
