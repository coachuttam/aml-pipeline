"""Offline demo: Ollama client, Modelfile, note writer, dev-only quant check."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src import config, ollama_generate, student_generate
from src.quant_check import agreement, compare, refuse_non_dev, verdict

MODELFILE = config.PROJECT_ROOT / "ollama" / "Modelfile"
NOTE = {"typology": "FAN_IN", "key_accounts": ["S"], "cited_txn_ids": [1],
        "red_flags": ["S pays R (txn 1)."], "investigator_questions": [],
        "reason": "One payment."}  # fmt: skip
TXN = {"transaction_id": 1, "hop": 0, "time": "2022-09-09 00:03", "from_account": "S",
       "to_account": "R", "amount_paid": 5.0, "payment_currency": "US Dollar",
       "amount_received": 5.0, "receiving_currency": "US Dollar",
       "payment_format": "ACH"}  # fmt: skip
EVIDENCE = {"alert_id": "A1", "alert_transaction": TXN, "linked_transactions": []}


def reply(text: str, reason: str = "stop") -> dict:
    return {"message": {"content": text}, "prompt_eval_count": 6000,
            "eval_count": 900, "done_reason": reason}  # fmt: skip


def modelfile_parameters() -> dict:
    lines = MODELFILE.read_text().splitlines()
    return {line.split()[1]: line.split(None, 2)[2] for line in lines
            if line.startswith("PARAMETER ") and "stop" not in line}  # fmt: skip


def test_the_modelfile_decodes_like_the_adapter_greedy_without_repeat_penalty():
    settings = modelfile_parameters()
    assert settings["temperature"] == "0" and settings["repeat_penalty"] == "1.0"
    assert int(settings["num_predict"]) == student_generate.MAX_NEW_TOKENS
    assert int(settings["num_ctx"]) >= 8192


def test_the_modelfile_uses_qwens_chat_format_and_the_gguf_named_in_config():
    text = MODELFILE.read_text()
    assert "FROM ../models/%s" % config.OLLAMA_GGUF in text
    assert "<|im_start|>{{ .Role }}" in text and 'stop "<|im_end|>"' in text
    assert "\nSYSTEM" not in text  # the caller always sends the v4 system prompt


def test_a_first_request_sends_no_options_and_a_retry_the_penalty():
    assert "options" not in ollama_generate.request_body([], "m", None)
    body = ollama_generate.request_body([], "m", 1.1)
    assert body["options"] == {"repeat_penalty": 1.1} and body["stream"] is False


def test_an_answer_cut_off_at_the_token_limit_is_marked_truncated():
    line = ollama_generate.answer_line("A1", reply("{", "length"), 3.0, False)
    assert line["truncated"] and not ollama_generate.is_valid(line)
    assert (line["input_tokens"], line["output_tokens"]) == (6000, 900)


def test_an_invalid_answer_is_retried_once_with_the_penalty(monkeypatch):
    calls = []

    def fake_chat(messages, model, host, penalty=None):
        calls.append(penalty)
        return reply("not json" if penalty is None else json.dumps(NOTE))

    monkeypatch.setattr(ollama_generate, "chat", fake_chat)
    line = ollama_generate.write_note({"alert_id": "A1", "messages": []})
    assert calls == [None, 1.1] and line["retried"] and ollama_generate.is_valid(line)


def test_an_ollama_note_goes_through_the_same_fact_check(monkeypatch):
    from src.note_writer import ollama_record

    bad = {**NOTE, "red_flags": ["S pays R (txn 7)."]}
    monkeypatch.setattr(ollama_generate, "chat",
                        lambda *a, **k: reply(json.dumps(bad)))  # fmt: skip
    record = ollama_record({"alert_id": "A1", "rank": 1, "score": 0.9}, EVIDENCE)
    assert record["violations"] == ["txn id 7 in text not in evidence"]
    assert record["note_version"] == config.OLLAMA_NOTE_VERSION


def test_the_ollama_writer_shows_the_adapters_saved_notes():
    assert config.NOTE_WRITERS["ollama"][0] == config.STUDENT_NOTES_FILE


def test_the_quantisation_check_refuses_anything_but_dev_answers(tmp_path: Path):
    refuse_non_dev(tmp_path / "dev_raw_q4km.jsonl")
    with pytest.raises(SystemExit):
        refuse_non_dev(tmp_path / "test_raw_Qwen_typology2.jsonl")


def record(typology: str) -> dict:
    return {"note": {**NOTE, "typology": typology}, "violations": [], "claims": 3}


def test_typology_agreement_counts_only_alerts_valid_in_both():
    first = {"A1": record("FAN_IN"), "A2": record("CYCLE"), "A3": record("RANDOM")}
    second = {"A1": record("FAN_IN"), "A2": record("FAN_OUT"),
              "A3": {"note": None, "violations": [], "claims": 0}}  # fmt: skip
    result = agreement(first, second)
    assert (result["both_valid"], result["same_typology"]) == (2, 1)


def test_the_pass_rules_use_the_thresholds_fixed_in_config():
    adapter = {"valid": 50, "invented_rate": 0.013}
    good = verdict(adapter, {"valid": 48, "invented_rate": 0.026},
                   {"typology_agreement": 0.8})  # fmt: skip
    assert all(good.values())
    bad = verdict(adapter, {"valid": 47, "invented_rate": 0.03},
                  {"typology_agreement": 0.79})  # fmt: skip
    assert not any(bad.values())


def test_comparing_two_dev_files_scores_each_against_its_evidence(tmp_path: Path):
    folder = tmp_path / "dev"
    (folder / config.EVIDENCE_FOLDER).mkdir(parents=True)
    pd.DataFrame([{"alert_id": "A1", "rank": 1, "score": 0.9}]).to_parquet(
        folder / config.ALERTS_FILE)  # fmt: skip
    (folder / config.EVIDENCE_FOLDER / "A1.json").write_text(json.dumps(EVIDENCE))
    for name, note in (
        ("dev_a.jsonl", NOTE),
        ("dev_b.jsonl", {**NOTE, "typology": "CYCLE"}),
    ):
        line = {"alert_id": "A1", "text": json.dumps(note), "input_tokens": 1,
                "output_tokens": 1}  # fmt: skip
        (tmp_path / name).write_text(json.dumps(line) + "\n")
    truth = [{"alert_id": "A1", "is_laundering": 1, "pattern_type": "FAN_IN",
              "pattern_visible": True}]  # fmt: skip
    report = compare(tmp_path / "dev_a.jsonl", tmp_path / "dev_b.jsonl", folder, truth)
    assert report["adapter"]["typology_right_visible"] == 1
    assert report["gguf_q4km"]["typology_right_visible"] == 0
    assert report["agreement"]["typology_agreement"] == 0.0
    assert report["passed"]["typology_agreement"] is False
