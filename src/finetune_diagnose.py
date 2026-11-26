"""Checks on the V2 data when the student fails: input format, targets, loss tokens.

1. `format_report`: every input file must hold the same system prompt and the same
   compact user-message shape as train (keys and their order). It also reports what
   may differ (hop mix, transactions per alert), which is content, not format.
2. `target_mix`: the assistant typology per split and kind of example.
3. `typology_token_share`: how much of the loss the typology value carries.

Example:
    format_report({"train": train_lines, "dev": dev_lines})
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Callable, Dict, List

from src import config


def user_shape(line: Dict) -> Dict:
    """Keys of the user message, of its transactions, and whether it is compact."""
    text = line["messages"][1]["content"]
    payload = json.loads(text)
    compact = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return {
        "keys": list(payload),
        "transaction_keys": sorted({tuple(t) for t in payload["transactions"]}),
        "compact": text == compact,
        "hops": Counter(t["hop"] for t in payload["transactions"]),
        "transactions": len(payload["transactions"]),
    }


def summarise_file(lines: List[Dict]) -> Dict:
    """Format facts (must match train) and content facts (may differ) of one file."""
    shapes = [user_shape(line) for line in lines]
    hops: Counter = Counter()
    for shape in shapes:
        hops.update(shape["hops"])
    total = sum(hops.values())
    return {
        "examples": len(lines),
        "system_prompts": sorted({line["messages"][0]["content"] for line in lines}),
        "user_keys": sorted({tuple(s["keys"]) for s in shapes}),
        "transaction_keys": sorted({k for s in shapes for k in s["transaction_keys"]}),
        "all_compact": all(s["compact"] for s in shapes),
        "hop_share": {str(h): round(n / total, 3) for h, n in sorted(hops.items())},
        "mean_transactions": round(
            sum(s["transactions"] for s in shapes) / len(shapes), 1
        ),
    }


def format_report(files: Dict[str, List[Dict]]) -> Dict:
    """Per-file summaries and, for each file, whether its format equals train's."""
    summaries = {name: summarise_file(lines) for name, lines in files.items()}
    reference = summaries["train"]
    format_keys = ("system_prompts", "user_keys", "transaction_keys", "all_compact")
    matches = {
        name: all(summary[key] == reference[key] for key in format_keys)
        for name, summary in summaries.items()
    }
    for name, summary in summaries.items():
        summary["format_matches_train"] = matches[name]
        summary["system_prompts"] = len(summary["system_prompts"])
        summary["user_keys"] = [list(keys) for keys in summary["user_keys"]]
        summary["transaction_keys"] = len(summary["transaction_keys"])
    return summaries


def example_kind(row: Dict) -> str:
    """false_alarm, visible (pattern shown) or hidden (laundering, thin evidence)."""
    if not row["is_laundering"]:
        return "false_alarm"
    return "visible" if row["pattern_visible"] else "hidden"


def target_mix(examples: Dict[str, List[Dict]], meta: List[Dict]) -> Dict:
    """Typology targets per split, overall and per kind of example."""
    kinds = {row["alert_id"]: example_kind(row) for row in meta}
    report = {}
    for split, lines in examples.items():
        overall: Counter = Counter()
        by_kind: Dict[str, Counter] = {}
        for line in lines:
            typology = json.loads(line["messages"][2]["content"])["typology"]
            overall[typology] += 1
            by_kind.setdefault(kinds[line["alert_id"]], Counter())[typology] += 1
        unclear = overall[config.UNCLEAR]
        report[split] = {
            "examples": len(lines),
            "unclear_share": round(unclear / len(lines), 3),
            "typology": dict(overall.most_common()),
            "by_kind": {k: dict(v.most_common()) for k, v in sorted(by_kind.items())},
        }
    return report


TYPOLOGY_PREFIX = '{"typology": "'


def typology_token_share(lines: List[Dict], count: Callable[[str], int]) -> Dict:
    """Share of loss tokens (assistant note + end token) in the typology value."""
    loss_tokens, value_tokens = 0, 0
    for line in lines:
        note = line["messages"][2]["content"]
        typology = json.loads(note)["typology"]
        loss_tokens += count(note) + 1  # +1: the end-of-turn token is also learned
        value_tokens += count(TYPOLOGY_PREFIX + typology + '"') - count(TYPOLOGY_PREFIX)
    return {
        "loss_tokens": loss_tokens,
        "mean_loss_tokens_per_example": round(loss_tokens / len(lines), 1),
        "typology_value_tokens": value_tokens,
        "typology_value_share": round(value_tokens / loss_tokens, 4),
    }


def load_token_counter() -> Dict:
    """Qwen's tokenizer if `tokenizers` is installed and reachable, else tiktoken."""
    try:
        from tokenizers import Tokenizer

        qwen = Tokenizer.from_pretrained(config.QWEN_TOKENIZER)
        return {
            "name": config.QWEN_TOKENIZER,
            "count": lambda t: len(qwen.encode(t).ids),
        }
    except Exception:  # not installed or offline: an estimate is still useful
        import tiktoken

        encoding = tiktoken.get_encoding(config.TOKEN_ENCODING)
        return {
            "name": config.TOKEN_ENCODING,
            "count": lambda t: len(encoding.encode(t)),
        }
