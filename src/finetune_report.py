"""Counts, typology mix, token lengths and teacher cost of a generated dataset.

Example:
    report = build_report(records, token_counts)    # dict saved as report.json
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

from src import config
from src.finetune_dataset import length_summary, teacher_cost


def typology_mix(records: List[Dict]) -> Dict[str, int]:
    """Kept notes by the typology they carry (false alarms count as unclear)."""
    return dict(Counter(r["note"]["typology"] for r in records).most_common())


def drop_reasons(records: List[Dict]) -> Dict[str, int]:
    """How often each kind of rejection happened (a note can have several)."""
    kinds = Counter()
    for record in records:
        for reason in record["rejected"]:
            kinds[reason.split(" not in")[0].split(" %")[0][:60]] += 1
    return dict(kinds.most_common())


def split_counts(records: List[Dict], kept: List[Dict]) -> Dict[str, Dict]:
    """Generated, kept and laundering counts per split."""
    result = {}
    for name in sorted({r["split"] for r in records}):
        made = [r for r in records if r["split"] == name]
        good = [r for r in kept if r["split"] == name]
        result[name] = {
            "generated": len(made),
            "kept": len(good),
            "kept_laundering": sum(r["is_laundering"] for r in good),
        }
    return result


def build_report(records: List[Dict], token_counts: Dict[str, List[int]]) -> Dict:
    """Everything to report about one generation run."""
    kept = [r for r in records if not r["rejected"]]
    return {
        "teacher_model": config.LLM_MODEL,
        "generated": len(records),
        "kept": len(kept),
        "dropped": len(records) - len(kept),
        "by_split": split_counts(records, kept),
        "typology_mix_kept": typology_mix(kept),
        "drop_reasons": drop_reasons(records),
        "tokens_per_example": {
            name: length_summary(counts) for name, counts in token_counts.items()
        },
        "teacher_cost_dollars": round(teacher_cost(records), 4),
    }
