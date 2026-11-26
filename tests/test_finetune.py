"""Tests for the fine-tuning dataset: selection, splits, teacher checks, files."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src import config
from src.case_note_input import build_blind_user_message
from src.finetune_alerts import build_pool, select_candidates
from src.finetune_dataset import (
    chat_example,
    length_summary,
    make_record,
    note_as_text,
    student_messages,
    teacher_cost,
)
from src.finetune_report import build_report
from src.finetune_select import assign_splits, pick_sample, select_alerts
from src.teacher import (
    build_teacher_message,
    privileged_truth,
    rejection_reasons,
)

TRANSACTION = {
    "transaction_id": 1, "from_account": "A", "to_account": "B", "hop": 0,
    "amount_paid": 100.0, "amount_received": 100.0, "payment_format": "ACH",
    "payment_currency": "US Dollar", "receiving_currency": "US Dollar",
    "time": "2022-09-01 00:00",
}  # fmt: skip
EVIDENCE = {
    "alert_transaction": TRANSACTION,
    "linked_transactions": [
        {**TRANSACTION, "transaction_id": i, "to_account": "C", "hop": 1}
        for i in (2, 3, 4)
    ],
    "counts": {"hop1_found": 3, "hop2_found": 0},
}
ALERT = {
    "alert_id": "A0001", "score": 0.9, "payment_format": "ACH",
    "payment_currency": "US Dollar",
    "reasons_json": json.dumps([{"feature": "sender_sent_7d", "value": 5, "shap": 1}]),
}  # fmt: skip


def alert_table() -> pd.DataFrame:
    """Made-up alerts: 6 laundering attempts of 5 alerts each, 40 false alarms."""
    rows = []
    for attempt in range(6):
        for turn in range(5):
            kind = "FAN_OUT" if attempt < 3 else "CYCLE"
            score = 0.9 - 0.01 * attempt - 0.001 * turn
            rows.append((1, kind, attempt, score, "L%d" % attempt))
    for index in range(40):
        rows.append((0, None, None, 0.5 - 0.001 * index, "F%d" % (index // 4)))
    table = pd.DataFrame(
        rows, columns=["is_laundering", "pattern_type", "attempt_id", "score",
                       "from_account"],
    )  # fmt: skip
    table["alert_id"] = ["X%03d" % i for i in range(len(table))]
    reasons = [{"feature": "payment_format_code", "value": 0.0, "shap": 1.0}]
    table["reasons_json"] = json.dumps(reasons)
    return table


@pytest.fixture
def small_selection(monkeypatch) -> pd.DataFrame:
    monkeypatch.setattr(config, "FINETUNE_ALERTS", 20)
    monkeypatch.setattr(config, "FINETUNE_ALERTS_PER_PATTERN", 3)
    monkeypatch.setattr(config, "FINETUNE_FALSE_ALARMS_PER_SENDER", 1)
    return select_alerts(alert_table())


def test_selection_keeps_the_laundering_share_and_caps_each_pattern(small_selection):
    laundering = small_selection[small_selection["is_laundering"] == 1]
    assert len(small_selection) == 20 and len(laundering) == 12
    assert laundering.groupby("attempt_id").size().max() <= 3


def test_selection_takes_one_alert_per_sender_when_capped_at_one(small_selection):
    false_alarms = small_selection[small_selection["is_laundering"] == 0]
    assert false_alarms["from_account"].is_unique


def test_no_pattern_attempt_is_in_both_train_and_validation(small_selection):
    split = assign_splits(small_selection)
    laundering = split[split["is_laundering"] == 1]
    sides = laundering.groupby("attempt_id")["split"].nunique()
    assert (sides == 1).all()
    assert set(laundering["split"]) == {"train", "validation"}


def test_sample_keeps_the_mix_and_spreads_the_typologies(small_selection):
    sample = pick_sample(small_selection, 10)
    assert len(sample) == 10 and sample["is_laundering"].sum() == 6
    assert set(sample["pattern_type"].dropna()) == {"FAN_OUT", "CYCLE"}


def test_a_laundering_alert_with_enough_visible_pattern_ids_uses_its_typology():
    row = {"is_laundering": 1, "pattern_type": "CYCLE",
           "pattern_edge_ids": json.dumps([1, 2, 3, 99])}  # fmt: skip
    truth = privileged_truth(row, EVIDENCE)
    assert truth["typology_to_use"] == "CYCLE"
    assert truth["pattern_transaction_ids_in_evidence"] == [1, 2, 3]


def test_thin_evidence_and_false_alarms_get_the_unclear_typology():
    thin = {"is_laundering": 1, "pattern_type": "CYCLE",
            "pattern_edge_ids": json.dumps([1, 98, 99])}  # fmt: skip
    clean = {"is_laundering": 0, "pattern_type": None, "pattern_edge_ids": "[]"}
    assert privileged_truth(thin, EVIDENCE)["typology_to_use"] == config.UNCLEAR
    assert privileged_truth(clean, EVIDENCE) == {
        "is_laundering": False, "typology_to_use": config.UNCLEAR,
    }  # fmt: skip


def test_student_input_never_holds_the_truth_but_the_teacher_input_does():
    student = student_messages(ALERT, EVIDENCE)
    truth = {"is_laundering": True, "typology_to_use": "CYCLE"}
    teacher = build_teacher_message(student[1]["content"], truth)
    assert teacher.startswith(student[1]["content"])
    assert "PRIVILEGED_TRUTH" in teacher
    text = " ".join(m["content"] for m in student).lower()
    assert "privileged" not in text and "is_laundering" not in text


def good_note(**changes) -> dict:
    note = {
        "typology": "CYCLE", "key_accounts": ["A"], "cited_txn_ids": [1, 2, 3],
        "red_flags": ["A paid B and C (txn 1, 2, 3)"],
        "investigator_questions": ["Why (txn 1)?", "Who (txn 2)?", "When (txn 3)?"],
        "reason": "Money moves in a ring.",
    }  # fmt: skip
    note.update(changes)
    return note


TRUTH = {"is_laundering": True, "typology_to_use": "CYCLE",
         "pattern_transaction_ids_in_evidence": [1, 2, 3]}  # fmt: skip


def test_a_note_that_passes_every_check_is_kept():
    assert rejection_reasons(good_note(), EVIDENCE, TRUTH) == []


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({"cited_txn_ids": [1, 2, 3, 77]}, "77"),
        ({"typology": "FAN_IN"}, "typology"),
        ({"investigator_questions": ["Only one (txn 1)?"]}, "questions"),
        ({"cited_txn_ids": [4]}, "pattern transactions"),
        ({"reason": "Ground truth says so."}, "ground truth"),
    ],
)
def test_a_bad_note_is_rejected_with_a_reason(changes, expected):
    reasons = rejection_reasons(good_note(**changes), EVIDENCE, TRUTH)
    assert any(expected in reason for reason in reasons)


def test_a_missing_note_is_rejected():
    assert rejection_reasons(None, EVIDENCE, TRUTH)


def test_chat_example_ends_with_the_note_in_schema_order():
    example = chat_example(student_messages(ALERT, EVIDENCE), good_note())
    roles = [m["role"] for m in example["messages"]]
    assert roles == ["system", "user", "assistant"]
    written = json.loads(note_as_text({**good_note(), "totals_by_currency": {}}))
    assert list(written) == [
        "typology", "key_accounts", "cited_txn_ids", "red_flags",
        "investigator_questions", "reason",
    ]  # fmt: skip


def test_length_summary_gives_median_and_95th_percentile():
    summary = length_summary(list(range(1, 101)))
    assert summary["p50"] == 50 and summary["p95"] == 95 and summary["max"] == 100


def test_teacher_cost_follows_the_config_price_list():
    records = [{"input_tokens": 1_000_000, "output_tokens": 1_000_000}]
    expected = config.LLM_PRICE_INPUT_PER_MILLION + config.LLM_PRICE_OUTPUT_PER_MILLION
    assert teacher_cost(records) == pytest.approx(expected)


def fake_client(note: dict):
    """An OpenAI-like client that always answers with `note`."""
    reply = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(note)))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
    )
    create = lambda **_: reply  # noqa: E731
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def test_make_record_keeps_a_good_teacher_note_and_drops_a_bad_one(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "LLM_CACHE_DIR", tmp_path)
    row = {"split": "train", "is_laundering": 1, "pattern_type": "CYCLE", "edge_id": 5,
           "pattern_edge_ids": json.dumps([1, 2, 3])}  # fmt: skip
    kept = make_record(fake_client(good_note()), ALERT, EVIDENCE, row)
    monkeypatch.setattr(config, "LLM_CACHE_DIR", tmp_path / "other")  # new cache
    dropped = make_record(
        fake_client(good_note(typology="STACK")), ALERT, EVIDENCE, row
    )
    assert kept["rejected"] == [] and dropped["rejected"]
    assert kept["input_tokens"] == 100 and kept["is_laundering"] == 1


def test_report_counts_kept_dropped_and_typologies():
    kept = {"split": "train", "is_laundering": 1, "note": good_note(), "rejected": [],
            "input_tokens": 10, "output_tokens": 5, "reused": False}  # fmt: skip
    bad = {**kept, "rejected": ["typology FAN_IN, wanted CYCLE"]}
    report = build_report([kept, bad], {"train": [100, 200]})
    assert report["kept"] == 1 and report["dropped"] == 1
    assert report["typology_mix_kept"] == {"CYCLE": 1}
    assert report["by_split"]["train"]["generated"] == 2


def test_finetune_data_comes_from_hi_small_only():
    assert "Small_HI" in str(config.HI_RAW_CSV)
    assert "Small_LI" not in str(config.HI_RAW_CSV)
    assert "Small_HI" in str(config.HI_PATTERNS_FILE)


def write_patterns(raw_csv: Path, path: Path) -> None:
    """A made-up patterns file: the late laundering rows, in two attempts."""
    lines = raw_csv.read_text().splitlines()[1:]
    late = [line for line in lines[len(lines) * 6 // 10 :] if line.endswith(",1")]
    half = len(late) // 2
    out = []
    for kind, group in (("FAN-OUT", late[:half]), ("CYCLE", late[half:])):
        out += ["BEGIN LAUNDERING ATTEMPT - %s:  Max 3 hops" % kind, *group]
        out.append("END LAUNDERING ATTEMPT - %s" % kind)
    path.write_text("\n".join(out) + "\n")


def test_candidates_hold_no_label_and_split_whole_patterns(
    tiny_csv: Path, tmp_path: Path, monkeypatch
) -> None:
    patterns = tmp_path / "HI-Small_Patterns.txt"
    write_patterns(tiny_csv, patterns)
    for name, value in {
        "MAX_TREES": 20, "FINETUNE_POOL": 300, "FINETUNE_ALERTS": 30,
        "FINETUNE_ALERTS_PER_PATTERN": 4, "HI_PATTERNS_FILE": patterns,
        "HI_FEATURE_CACHE": tmp_path / "features",
    }.items():  # fmt: skip
        monkeypatch.setattr(config, name, value)
    pool = build_pool(tmp_path / "pool", tiny_csv)
    dev = tmp_path / "dev"
    dev.mkdir()
    pd.read_parquet(pool / "alerts.parquet").head(10).to_parquet(dev / "alerts.parquet")
    folder = select_candidates(pool, tmp_path / "candidates", dev, tiny_csv)
    alerts = pd.read_parquet(folder / "alerts.parquet")
    truth = pd.read_parquet(folder / "ground_truth.parquet")
    assert not [c for c in alerts.columns if "laundering" in c or "pattern" in c]
    assert alerts["alert_id"].iloc[0] == "A0001"
    assert len(list((folder / "evidence").glob("*.json"))) == len(alerts)
    laundering = truth[truth["is_laundering"] == 1]
    assert len(laundering) > 0
    assert (laundering.groupby("attempt_id")["split"].nunique() == 1).all()


def test_student_input_is_the_blind_teacher_input_written_compactly():
    student = student_messages(ALERT, EVIDENCE)[1]["content"]
    assert "\n" not in student and ", " not in student
    assert json.loads(student) == json.loads(build_blind_user_message(ALERT, EVIDENCE))
