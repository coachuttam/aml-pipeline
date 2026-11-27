"""Final results table and chart, and the student as the V2 note writer (made-up)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import config
from src.final_results import draw_chart, result_rows, results_markdown
from src.note_writer import generate_command, typology_text, write_notes


def measure(count: int, of: int) -> dict:
    return {"count": count, "of": of, "rate": count / of}


def writer_scores(typology: int, unclear: int, invented: int) -> dict:
    return {"typology_accuracy_visible": measure(typology, 37),
            "unclear_rate_false_alarms": measure(unclear, 147),
            "invented_facts_per_claim": measure(invented, 6000),
            "json_validity": measure(200, 200)}  # fmt: skip


REPORT = {
    "student": writer_scores(18, 136, 6),
    "v1_prompt_v3": writer_scores(12, 0, 20),
    "rules": {"typology_right_visible": 7, "visible": 37,
              "unclear_false_alarms": 72, "false_alarms": 147},
    "always_gather_scatter": {"typology_right_visible": 10, "visible": 37,
                              "unclear_false_alarms": 0, "false_alarms": 147},
}  # fmt: skip


def test_the_results_table_has_one_row_per_system_with_counts_and_rates():
    table = results_markdown(result_rows(REPORT))
    assert "18/37 (48.6%)" in table and "6/6000 (1.0 per 1,000)" in table
    assert table.count("\n") == 5  # header, rule, four systems


def test_baselines_have_no_invented_facts_because_they_write_no_text():
    rules = result_rows(REPORT)[2]
    assert rules["invented"] is None and "n/a" in results_markdown([rules])


def test_the_chart_is_written_as_a_png(tmp_path: Path):
    draw_chart(result_rows(REPORT), tmp_path / "chart.png")
    assert (tmp_path / "chart.png").read_bytes()[:4] == b"\x89PNG"


def test_a_typology_is_shown_as_a_suggestion_the_analyst_confirms():
    text = typology_text({"typology": "FAN_IN"}, "student")
    assert text.startswith("Suggested typology: FAN_IN") and "analyst confirms" in text
    assert "does not show" in typology_text({"typology": "unclear"}, "student")


def test_generation_uses_the_frozen_adapter_from_config(tmp_path: Path):
    command = generate_command(tmp_path / "in.jsonl", tmp_path / "out.jsonl")
    assert command[command.index("--adapter") + 1] == str(config.STUDENT_ADAPTER_DIR)


def test_student_answers_become_fact_checked_notes_without_ground_truth(tmp_path):
    txn = {"transaction_id": 1, "hop": 0, "time": "2022-09-07 10:00",
           "from_account": "S", "to_account": "R", "amount_paid": 5.0,
           "payment_currency": "Yen", "amount_received": 5.0,
           "receiving_currency": "Yen", "payment_format": "ACH"}  # fmt: skip
    evidence = {"alert_id": "A1", "alert_transaction": txn, "linked_transactions": []}
    pd.DataFrame([{"alert_id": "A1", "rank": 1, "score": 1.0}]).to_parquet(
        tmp_path / config.ALERTS_FILE
    )
    (tmp_path / config.EVIDENCE_FOLDER).mkdir()
    (tmp_path / config.EVIDENCE_FOLDER / "A1.json").write_text(json.dumps(evidence))
    note = {"typology": "unclear", "key_accounts": ["S"], "cited_txn_ids": [1],
            "red_flags": ["S pays R (txn 9)."], "investigator_questions": [],
            "reason": ""}  # fmt: skip
    raw = tmp_path / "raw.jsonl"
    answer = {"alert_id": "A1", "text": json.dumps(note),
              "input_tokens": 1, "output_tokens": 1}  # fmt: skip
    raw.write_text(json.dumps(answer) + "\n")
    assert write_notes(tmp_path, raw) == {"alerts": 1, "needs_review": 1}
    saved = json.loads((tmp_path / config.STUDENT_NOTES_FILE).read_text())
    assert saved[0]["violations"] == ["txn id 9 in text not in evidence"]
