"""Rebalancing the train split and scoring the student's raw answers (no GPU)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from src import config
from src.finetune_rebalance import example_kind, rebalance_ids
from src.score_student import student_record

ROOT = Path(__file__).resolve().parent.parent


def meta_row(alert_id, kind, split="train") -> dict:
    row = {"alert_id": alert_id, "split": split, "is_laundering": 0,
           "typology_to_use": "unclear"}  # fmt: skip
    if kind == "typed":
        row.update(is_laundering=1, typology_to_use="CYCLE")
    if kind == "thin":
        row.update(is_laundering=1)
    return row


ROWS = (
    [meta_row("t%d" % i, "typed") for i in range(60)]
    + [meta_row("n%d" % i, "thin") for i in range(20)]
    + [meta_row("f%d" % i, "false") for i in range(150)]
    + [meta_row("v%d" % i, "false", "validation") for i in range(10)]
)


def test_rebalancing_keeps_all_laundering_and_makes_it_half_of_train():
    kept, missing = rebalance_ids(ROWS)
    assert sum(i[0] in "tn" for i in kept) == 80  # every typed and thin example
    assert sum(i.startswith("f") for i in kept) == 80  # 80 / (80 + 80) = 0.5
    assert missing == 0 and not [i for i in kept if i.startswith("v")]


def test_rebalancing_reports_how_many_false_alarms_were_missing():
    few = [r for r in ROWS if not r["alert_id"].startswith("f")][:80]
    few += [meta_row("f%d" % i, "false") for i in range(50)]
    kept, missing = rebalance_ids(few)
    assert sum(i.startswith("f") for i in kept) == 50 and missing == 30


def test_rebalancing_is_seeded_and_repeatable():
    assert rebalance_ids(ROWS) == rebalance_ids(ROWS)


def test_example_kind_names_the_three_kinds():
    assert example_kind(meta_row("a", "typed")) == "typed"
    assert example_kind(meta_row("a", "thin")) == "thin_laundering"
    assert example_kind(meta_row("a", "false")) == "false_alarm"


EVIDENCE = {
    "alert_transaction": {
        "transaction_id": 1, "from_account": "A", "to_account": "B",
        "amount_paid": 100.0, "amount_received": 100.0,
        "payment_currency": "US Dollar",
    },
    "linked_transactions": [],
}  # fmt: skip
ALERT = {"alert_id": "A0001", "rank": 1, "score": 0.9}
NOTE = {
    "typology": "unclear", "key_accounts": ["A"], "cited_txn_ids": [1],
    "red_flags": ["A paid B (txn 1)"], "investigator_questions": ["q (txn 1)"] * 3,
    "reason": "Thin evidence.",
}  # fmt: skip


def raw(text: str) -> dict:
    return {"text": text, "input_tokens": 10, "output_tokens": 5}


def test_a_good_student_answer_becomes_a_checked_record_with_totals():
    record = student_record(ALERT, EVIDENCE, raw(json.dumps(NOTE)))
    assert record["note"]["totals_by_currency"] == {"USD": 100.0}
    assert record["violations"] == [] and not record["needs_review"]


def test_an_answer_citing_an_unknown_transaction_is_flagged_for_review():
    bad = {**NOTE, "cited_txn_ids": [1, 99]}
    record = student_record(ALERT, EVIDENCE, raw(json.dumps(bad)))
    assert record["needs_review"] and record["violations"]


def test_text_that_is_not_json_or_lacks_a_field_has_no_note():
    for text in ("I think this is fan-out.", json.dumps({"typology": "CYCLE"}), ""):
        assert student_record(ALERT, EVIDENCE, raw(text))["note"] is None


def test_the_gpu_script_and_notebook_are_valid_python_and_use_the_same_model():
    script = (ROOT / "src" / "student_generate.py").read_text()
    ast.parse(script)
    notebook = json.loads((ROOT / "notebooks" / "finetune_colab.ipynb").read_text())
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    for cell in code:
        lines = [line for line in "".join(cell["source"]).splitlines()
                 if not line.lstrip().startswith(("!", "%"))]  # fmt: skip
        ast.parse("\n".join(lines))
    assert "Qwen/Qwen2.5-7B-Instruct" in script
    assert "Qwen/Qwen2.5-7B-Instruct" in "".join(code[1]["source"])
    assert config.STUDENT_PROMPT_VERSION == "v4"
