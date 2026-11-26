"""V2 diagnosis: rule-based typologies, the HI-Small val set, format and loss checks."""

from __future__ import annotations

import json

from src import config
from src.finetune_diagnose import format_report, target_mix, typology_token_share
from src.finetune_val import pick_val_alerts, round_robin_by_type
from src.typology_rules import RuleSettings, detect_typology, score_rules, tune

LOOSE = RuleSettings(window_hours=24, min_degree=3, hub_min_payers=3)


def payment(transaction_id: int, sender: str, receiver: str, minute: int = 0) -> dict:
    time = "2022-09-01 10:%02d" % minute
    return {"transaction_id": transaction_id, "hop": 1, "time": time,
            "from_account": sender, "to_account": receiver}  # fmt: skip


def evidence(alert: dict, linked: list) -> dict:
    return {"alert_transaction": {**alert, "hop": 0}, "linked_transactions": linked}


def test_money_coming_back_to_the_sender_is_a_cycle():
    case = evidence(payment(1, "S", "R"), [payment(2, "R", "X"), payment(3, "X", "S")])
    assert detect_typology(case, LOOSE) == "CYCLE"


def test_one_sender_paying_many_is_fan_out():
    linked = [payment(i, "S", "P%d" % i) for i in range(2, 5)]
    assert detect_typology(evidence(payment(1, "S", "R"), linked), LOOSE) == "FAN_OUT"


def test_many_paying_the_receiver_is_fan_in():
    linked = [payment(i, "P%d" % i, "R") for i in range(2, 5)]
    assert detect_typology(evidence(payment(1, "S", "R"), linked), LOOSE) == "FAN_IN"


def test_a_receiver_that_gathers_and_scatters_is_gather_scatter():
    linked = [payment(i, "P%d" % i, "R") for i in range(2, 5)]
    linked += [payment(i, "R", "Q%d" % i) for i in range(5, 8)]
    assert (
        detect_typology(evidence(payment(1, "S", "R"), linked), LOOSE)
        == "GATHER_SCATTER"
    )


def test_scatter_to_middles_that_all_pay_one_sink_is_scatter_gather():
    linked = [payment(i, "S", "M%d" % i) for i in range(2, 4)]
    linked += [payment(10 + i, "M%d" % i, "SINK") for i in range(2, 4)] + [
        payment(9, "R", "SINK")
    ]
    assert (
        detect_typology(evidence(payment(1, "S", "R"), linked), LOOSE)
        == "SCATTER_GATHER"
    )


def test_transactions_outside_the_window_are_ignored_and_give_unclear():
    far = [
        {**payment(i, "S", "P%d" % i), "time": "2022-09-20 10:00"} for i in range(2, 5)
    ]
    assert detect_typology(evidence(payment(1, "S", "R"), far), LOOSE) == config.UNCLEAR


def test_tuning_picks_the_settings_with_the_most_right_typologies():
    linked = [payment(i, "S", "P%d" % i) for i in range(2, 5)]
    cases = {"A1": evidence(payment(1, "S", "R"), linked)}
    truth = [{"alert_id": "A1", "is_laundering": 1, "pattern_visible": True,
              "pattern_type": "FAN_OUT"}]  # fmt: skip
    grid = [RuleSettings(24, 5, 5), LOOSE]
    settings, result = tune(cases, truth, grid)
    assert settings == LOOSE and result["typology_right_visible"] == 1
    assert score_rules({"A1": "unclear"}, truth)["typology_right_visible"] == 0


def meta_row(
    alert_id, laundering, visible, pattern="FAN_IN", split="validation"
) -> dict:
    return {"alert_id": alert_id, "split": split, "is_laundering": laundering,
            "pattern_visible": visible, "pattern_type": pattern}  # fmt: skip


def test_round_robin_spreads_typologies_and_prefers_new_pattern_attempts():
    rows = [
        {"pattern_type": "FAN_IN", "attempt_id": 1, "alert_id": "a%d" % i}
        for i in range(5)
    ]
    rows += [
        {"pattern_type": "CYCLE", "attempt_id": 2 + i, "alert_id": "c%d" % i}
        for i in range(5)
    ]
    picked = round_robin_by_type(rows, 4, seed=0)
    assert sorted(r["pattern_type"] for r in picked) == [
        "CYCLE",
        "CYCLE",
        "FAN_IN",
        "FAN_IN",
    ]
    assert len({r["attempt_id"] for r in picked if r["pattern_type"] == "FAN_IN"}) == 1


def test_val_set_uses_validation_only_with_the_dev_mix():
    meta = [meta_row("v%d" % i, 1, True) for i in range(30)]
    meta += [meta_row("f%d" % i, 0, False) for i in range(30)]
    meta += [meta_row("h%d" % i, 1, False) for i in range(30)]
    meta += [meta_row("t%d" % i, 1, True, split="train") for i in range(30)]
    truth = [{"alert_id": m["alert_id"], "attempt_id": i} for i, m in enumerate(meta)]
    chosen = pick_val_alerts(meta, truth)
    assert len(chosen) == 50 and not [i for i in chosen if i.startswith("t")]
    assert sum(i.startswith("v") for i in chosen) == config.VAL_SET_VISIBLE


def chat_line(alert_id: str, user: dict, typology: str, system: str = "rules") -> dict:
    note = json.dumps({"typology": typology, "reason": "short"})
    return {"alert_id": alert_id, "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, separators=(",", ":"))},
        {"role": "assistant", "content": note}]}  # fmt: skip


USER = {"alert_transaction_id": 1, "transactions": [{"transaction_id": 1, "hop": 0}]}


def test_format_report_flags_a_different_system_prompt_but_not_other_content():
    files = {"train": [chat_line("a", USER, "CYCLE")],
             "dev": [chat_line("b", USER, "unclear")],
             "odd": [chat_line("c", USER, "CYCLE", system="other")]}  # fmt: skip
    report = format_report(files)
    assert report["dev"]["format_matches_train"]
    assert not report["odd"]["format_matches_train"]


def test_target_mix_counts_unclear_per_kind_of_example():
    lines = {"train": [chat_line("a", USER, "CYCLE"), chat_line("b", USER, "unclear")]}
    meta = [meta_row("a", 1, True), meta_row("b", 0, False)]
    mix = target_mix(lines, meta)["train"]
    assert mix["unclear_share"] == 0.5 and mix["by_kind"]["false_alarm"] == {
        "unclear": 1
    }


def test_typology_share_counts_the_value_tokens_against_all_loss_tokens():
    count_words = lambda text: len(text.replace('"', " ").split())  # noqa: E731
    share = typology_token_share([chat_line("a", USER, "CYCLE")], count_words)
    assert share["typology_value_tokens"] == 1
    assert 0 < share["typology_value_share"] < 1
