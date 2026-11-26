"""Write student case notes with the fine-tuned LoRA adapter (needs a GPU).

Self-contained on purpose (no imports from this repo) so it can be copied to Google
Drive and run in Colab next to the notebook:

    python student_generate.py --inputs dev_inputs.jsonl --output dev_raw.jsonl \
        --adapter /content/drive/MyDrive/aml_finetune/adapter

Each input line is {"alert_id", "messages": [system, user]}. Each output line is
{"alert_id", "text", "input_tokens", "output_tokens"}: the raw answer, scored later
on a CPU by `python -m src.score_student`. Greedy decoding (temperature 0), batches
of similar length, and lines already in the output file are skipped, so a Colab
disconnect loses little and a finished alert is never generated twice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # must be the model the adapter was trained on
MAX_NEW_TOKENS = (
    1400  # teacher notes: p99 1,343 tokens; an answer cut off here is invalid
)
RETRY_REPETITION_PENALTY = 1.1  # used once for an invalid answer, never the first time
NOTE_FIELDS = {"typology", "key_accounts", "cited_txn_ids", "red_flags",
               "investigator_questions", "reason"}  # fmt: skip
BATCH_SIZE = 4  # decoding is memory-bound, so a few prompts at once are nearly free


def load_lines(path: Path) -> List[Dict]:
    """The JSON lines of a file, or none if it does not exist."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_model(adapter: str, base_model: str):
    """Base model in 4-bit (as trained) with the LoRA adapter on top."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )  # fmt: skip
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=quantization, device_map="auto"
    )
    return PeftModel.from_pretrained(model, adapter).eval(), tokenizer


def generate_batch(
    model, tokenizer, batch: List[Dict], repetition_penalty: float = 1.0
) -> List[Dict]:
    """Answers for one batch of inputs (greedy; a penalty above 1 only on a retry)."""
    import torch

    texts = [
        tokenizer.apply_chat_template(
            item["messages"], tokenize=False, add_generation_prompt=True
        )
        for item in batch
    ]
    encoded = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            repetition_penalty=repetition_penalty, pad_token_id=tokenizer.pad_token_id,
        )  # fmt: skip
    answers = output[:, encoded["input_ids"].shape[1] :]
    stop = model.generation_config.eos_token_id or tokenizer.eos_token_id
    stop_ids = set(stop if isinstance(stop, list) else [stop])
    return [
        {
            "alert_id": item["alert_id"],
            "text": tokenizer.decode(answer, skip_special_tokens=True),
            "input_tokens": int(mask.sum()),
            "output_tokens": int((answer != tokenizer.pad_token_id).sum()),
            "truncated": not any(int(t) in stop_ids for t in answer),
        }
        for item, answer, mask in zip(batch, answers, encoded["attention_mask"])
    ]


def is_valid(line: Dict) -> bool:
    """An answer that stopped by itself and is JSON with exactly the six fields."""
    if line.get("truncated"):
        return False  # cut off at MAX_NEW_TOKENS: invalid, never repaired
    try:
        note = json.loads(line["text"])
    except ValueError:
        return False
    return isinstance(note, dict) and set(note) == NOTE_FIELDS


def needs_retry(line: Dict) -> bool:
    """An invalid answer that has not been retried yet."""
    return not is_valid(line) and not line.get("retried")


def run_pass(model, tokenizer, items, size, handle, penalty=1.0, retried=False):
    """Generate for `items` in batches and append each answer to the output file."""
    for start in range(0, len(items), size):
        batch = items[start : start + size]
        for line in generate_batch(model, tokenizer, batch, penalty):
            handle.write(json.dumps({**line, "retried": retried}) + "\n")
        handle.flush()
        print("%d / %d" % (min(start + size, len(items)), len(items)), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    arguments = parser.parse_args()
    inputs = load_lines(arguments.inputs)
    inputs.sort(key=lambda i: len(i["messages"][1]["content"]))
    done = {line["alert_id"]: line for line in load_lines(arguments.output)}
    model, tokenizer = load_model(arguments.adapter, arguments.base_model)
    with arguments.output.open("a") as handle:
        todo = [i for i in inputs if i["alert_id"] not in done]
        run_pass(model, tokenizer, todo, arguments.batch_size, handle)
        answers = {a["alert_id"]: a for a in load_lines(arguments.output)}
        retry = [i for i in inputs if needs_retry(answers[i["alert_id"]])]
        print("retrying %d invalid answers once" % len(retry), flush=True)
        run_pass(model, tokenizer, retry, arguments.batch_size, handle,
                 RETRY_REPETITION_PENALTY, retried=True)  # fmt: skip


if __name__ == "__main__":
    main()
