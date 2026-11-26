"""train_v3: balanced typologies, 35% "unclear" from the right rows, the val gate."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from src import config
from src.finetune_v3 import (
    build_train_v3,
    cap_or_repeat,
    l4_hours,
    pick_v3,
    typology_first,
    val_gate,
)


def row(alert_id: str, typology: str, laundering: int, split: str = "train") -> dict:
    visible = typology != config.UNCLEAR
    return {"alert_id": alert_id, "split": split, "is_laundering": laundering,
            "typology_to_use": typology, "pattern_type": typology,
            "pattern_visible": visible, "attempt_id": "P" + alert_id}  # fmt: skip


def made_up_rows() -> list:
    rows = [row("F%03d" % i, "FAN_OUT", 1) for i in range(30)]
    rows += [row("C%03d" % i, "CYCLE", 1) for i in range(5)]
    rows += [row("A%03d" % i, config.UNCLEAR, 0) for i in range(40)]
    rows += [row("H%03d" % i, config.UNCLEAR, 1) for i in range(40)]
    rows += [row("V%03d" % i, "FAN_OUT", 1, split="validation") for i in range(5)]
    return rows


@pytest.fixture
def ten_per_typology(monkeypatch):
    monkeypatch.setattr(config, "FINETUNE_V3_PER_TYPOLOGY", 10)


def test_a_big_class_is_capped_without_copies():
    rows = [row("F%03d" % i, "FAN_OUT", 1) for i in range(30)]
    picked = cap_or_repeat(rows, 10, seed=0)
    assert len(picked) == 10 and len({r["alert_id"] for r in picked}) == 10


def test_a_small_class_keeps_every_example_and_adds_copies():
    rows = [row("C%03d" % i, "CYCLE", 1) for i in range(4)]
    picked = cap_or_repeat(rows, 10, seed=0)
    assert len(picked) == 10
    assert Counter(r["alert_id"] for r in picked).most_common()[-1][1] >= 2


def test_every_typology_gets_the_same_count(ten_per_typology):
    typed = Counter(r["typology_to_use"] for r in pick_v3(made_up_rows()))
    assert typed["FAN_OUT"] == typed["CYCLE"] == 10


def test_unclear_is_about_35_percent_and_only_false_alarms_or_thin_laundering(
    ten_per_typology,
):
    chosen = pick_v3(made_up_rows())
    unclear = [r for r in chosen if r["typology_to_use"] == config.UNCLEAR]
    assert abs(len(unclear) / len(chosen) - 0.35) < 0.02
    assert all(not r["pattern_visible"] for r in unclear)
    assert Counter(r["is_laundering"] for r in unclear)[0] > len(unclear) / 2


def test_validation_examples_never_enter_train_v3(ten_per_typology):
    assert all(r["split"] == "train" for r in pick_v3(made_up_rows()))


def test_typology_is_moved_to_the_first_field():
    note = json.dumps({"reason": "r", "typology": "CYCLE", "red_flags": []})
    assert list(json.loads(typology_first(note))) == ["typology", "reason", "red_flags"]


def test_a_note_that_starts_with_typology_is_left_byte_for_byte():
    note = '{"typology": "CYCLE", "reason": "r"}'
    assert typology_first(note) == note


def test_build_writes_train_v3_from_train_full_and_meta(tmp_path, ten_per_typology):
    rows = made_up_rows()
    meta = [{k: v for k, v in r.items() if k != "attempt_id"} for r in rows]
    (tmp_path / "meta.jsonl").write_text("".join(json.dumps(r) + "\n" for r in meta))
    examples = []
    for r in rows:
        note = json.dumps({"reason": "r", "typology": r["typology_to_use"]})
        messages = [{"role": "system", "content": "s"},
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": note}]  # fmt: skip
        examples.append({"alert_id": r["alert_id"], "messages": messages})
    text = "".join(json.dumps(e) + "\n" for e in examples)
    (tmp_path / "train_full.jsonl").write_text(text)
    report = build_train_v3(tmp_path, {})
    lines = (tmp_path / config.FINETUNE_V3_TRAIN).read_text().splitlines()
    assert len(lines) == report["examples"]
    first_fields = {list(json.loads(json.loads(x)["messages"][2]["content"]))[0]
                    for x in lines}  # fmt: skip
    assert first_fields == {"typology"}


def gate_report(typology_correct: int, unclear_false_alarms: int) -> dict:
    return {"typology_accuracy_visible": {"count": typology_correct, "of": 20},
            "unclear_rate_false_alarms": {"count": unclear_false_alarms, "of": 20},
            "false_alarm_alerts": 20}  # fmt: skip


def test_the_gate_passes_at_10_typologies_and_half_the_false_alarms_unclear():
    assert val_gate(gate_report(10, 10))["passed"]


def test_the_gate_fails_the_first_student_that_always_said_unclear():
    assert not val_gate(gate_report(1, 18))["passed"]


def test_the_gate_fails_a_student_that_never_says_unclear_on_false_alarms():
    assert not val_gate(gate_report(15, 9))["passed"]


def test_two_epochs_take_twice_as_long_as_one():
    assert l4_hours(1_000_000, 0, 2) == pytest.approx(2 * l4_hours(1_000_000, 0, 1))
