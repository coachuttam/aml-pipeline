"""train_v3.jsonl: the final retrain set, balanced typologies and less "unclear".

The first student answered "unclear" for 18 of 20 visible-pattern validation alerts:
train was 63.5% "unclear" and GATHER_SCATTER had three times RANDOM's examples. This
set reuses the same examples (inputs and teacher notes unchanged) and only changes
the mix:

- each typology gets FINETUNE_V3_PER_TYPOLOGY examples: big classes are capped
  (new pattern attempts before repeats), small ones repeated (seeded copies);
- "unclear" is FINETUNE_V3_UNCLEAR_SHARE of the file and comes only from false
  alarms and laundering whose evidence is too thin to show the pattern;
- the typology is the first field of every note, so it is decided first.

The validation file and every test file are never read or changed here.

Example:
    chosen = pick_v3(rows, examples)   # rows: meta rows with attempt_id
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

from src import config
from src.finetune_diagnose import example_kind
from src.finetune_val import round_robin_by_type


def cap_or_repeat(rows: List[Dict], count: int, seed: int) -> List[Dict]:
    """`count` rows of one typology: a spread subset, or all of them plus copies."""
    if len(rows) >= count:
        return round_robin_by_type(rows, count, seed)
    copies = rows[:]
    random.Random(seed).shuffle(copies)
    return rows + [copies[i % len(copies)] for i in range(count - len(rows))]


def unclear_wanted(typed: int) -> int:
    """Unclear examples that make "unclear" the wanted share of the file."""
    share = config.FINETUNE_V3_UNCLEAR_SHARE
    return int(round(typed * share / (1 - share)))


def pick_unclear(rows: List[Dict], count: int, seed: int) -> List[Dict]:
    """False alarms (FINETUNE_V3_FALSE_ALARM_SHARE of them) and thin laundering."""
    alarms = sorted((r for r in rows if example_kind(r) == "false_alarm"),
                    key=lambda r: r["alert_id"])  # fmt: skip
    thin = sorted((r for r in rows if example_kind(r) == "hidden"),
                  key=lambda r: r["alert_id"])  # fmt: skip
    alarm_count = min(
        len(alarms), int(round(count * config.FINETUNE_V3_FALSE_ALARM_SHARE))
    )
    thin_count = min(len(thin), count - alarm_count)
    generator = random.Random(seed)
    return generator.sample(alarms, alarm_count) + generator.sample(thin, thin_count)


def pick_v3(rows: List[Dict]) -> List[Dict]:
    """Meta rows of train_v3 in file order (a typed row may appear more than once)."""
    train = [r for r in rows if r["split"] == "train"]
    typed: List[Dict] = []
    by_type: Dict[str, List[Dict]] = {}
    for row in train:
        if example_kind(row) == "visible":
            by_type.setdefault(row["typology_to_use"], []).append(row)
    for name in sorted(by_type):
        typed += cap_or_repeat(by_type[name], config.FINETUNE_V3_PER_TYPOLOGY,
                               config.FINETUNE_V3_SEED)  # fmt: skip
    unclear = [r for r in train if example_kind(r) != "visible"]
    chosen = typed + pick_unclear(unclear, unclear_wanted(len(typed)),
                                  config.FINETUNE_V3_SEED)  # fmt: skip
    random.Random(config.FINETUNE_V3_SEED).shuffle(chosen)  # copies not side by side
    return chosen


def typology_first(note_text: str) -> str:
    """The note with "typology" as its first JSON field (unchanged if it already is)."""
    note = json.loads(note_text)
    if list(note)[0] == "typology":
        return note_text
    ordered = {"typology": note["typology"]}
    ordered.update((k, v) for k, v in note.items() if k != "typology")
    return json.dumps(ordered, ensure_ascii=False)


def v3_examples(chosen: List[Dict], examples: Dict[str, Dict]) -> List[Dict]:
    """The chat examples of the chosen rows, each note starting with the typology."""
    lines = []
    for row in chosen:
        example = examples[row["alert_id"]]
        messages = [dict(m) for m in example["messages"]]
        messages[2]["content"] = typology_first(messages[2]["content"])
        lines.append({"alert_id": row["alert_id"], "messages": messages})
    return lines


def mix_report(chosen: List[Dict]) -> Dict:
    """Size, unclear share, typology counts and kinds of the chosen rows."""
    typology = Counter(r["typology_to_use"] for r in chosen)
    return {
        "examples": len(chosen),
        "distinct_alerts": len({r["alert_id"] for r in chosen}),
        "unclear_share": round(typology[config.UNCLEAR] / max(len(chosen), 1), 3),
        "typology": dict(typology.most_common()),
        "kinds": dict(Counter(example_kind(r) for r in chosen).most_common()),
    }


def build_train_v3(folder: Path, attempts: Dict[str, str]) -> Dict:
    """Write folder/train_v3.jsonl from train_full.jsonl and meta.jsonl; its report."""
    rows = [json.loads(line) for line in (folder / "meta.jsonl").open()]
    rows = [{**r, "pattern_type": r["typology_to_use"],
             "attempt_id": attempts.get(r["alert_id"])} for r in rows]  # fmt: skip
    examples = {}
    for line in (folder / "train_full.jsonl").open():
        example = json.loads(line)
        examples[example["alert_id"]] = example
    chosen = pick_v3([r for r in rows if r["alert_id"] in examples])
    lines = v3_examples(chosen, examples)
    text = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in lines)
    (folder / config.FINETUNE_V3_TRAIN).write_text(text)
    return mix_report(chosen)


def val_gate(report: Dict) -> Dict:
    """Pass only if the student names typologies and still hedges on false alarms."""
    typology_correct = report["typology_accuracy_visible"]["count"]
    alarms = report["false_alarm_alerts"]
    unclear = report["unclear_rate_false_alarms"]["count"] / alarms if alarms else 0.0
    passed = (
        typology_correct >= config.VAL_GATE_MIN_TYPOLOGY_CORRECT
        and unclear >= config.VAL_GATE_MIN_UNCLEAR_FALSE_ALARMS
    )
    return {
        "typology_correct": typology_correct,
        "unclear_share_false_alarms": round(unclear, 3),
        "passed": passed,
    }


def example_tokens(example: Dict, count) -> int:
    """Tokens of one chat example: its three messages plus the chat template."""
    text = sum(count(m["content"]) for m in example["messages"])
    return text + config.CHAT_TEMPLATE_TOKENS


def l4_hours(train_tokens: int, validation_tokens: int, epochs: int) -> float:
    """L4 training hours at L4_TOKENS_PER_SECOND, one validation pass per epoch."""
    seconds = epochs * (train_tokens + validation_tokens / config.EVAL_SPEEDUP)
    return round(seconds / config.L4_TOKENS_PER_SECOND / 3600, 1)
