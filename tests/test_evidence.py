"""Tests for src/evidence.py on a small hand-made graph."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src import config
from src.data import load_in_multignn_order
from src.evidence import EvidenceIndex, build_evidence, write_evidence
from tests.conftest import raw_line, write_raw

DAY = 24 * 60


def graph_dataset(tmp_path: Path, lines):
    """The label-free table of a tiny graph (too short to need a train/test split)."""
    frame = load_in_multignn_order(write_raw(tmp_path / "graph.csv", lines))
    return SimpleNamespace(transactions=frame.drop(columns=["is_laundering"]))


def row_of(dataset, edge_id: int) -> int:
    return int(
        dataset.transactions.index[dataset.transactions["edge_id"] == edge_id][0]
    )


def test_hop_one_is_other_payments_touching_the_sender_or_receiver(
    tmp_path: Path,
) -> None:
    lines = [
        raw_line(10 * DAY, "S", "R"),  # 0: the alert
        raw_line(10 * DAY + 5, "S", "X"),  # 1: sender pays another account
        raw_line(10 * DAY + 9, "Y", "R"),  # 2: someone pays the receiver
        raw_line(10 * DAY + 20, "P", "Q"),  # 3: unrelated
    ]
    dataset = graph_dataset(tmp_path, lines)
    document = build_evidence(
        EvidenceIndex(dataset.transactions), row_of(dataset, 0), "A0001", "nearest"
    )
    linked = {t["transaction_id"]: t for t in document["linked_transactions"]}
    assert set(linked) == {1, 2} and all(t["hop"] == 1 for t in linked.values())
    assert document["alert_transaction"]["transaction_id"] == 0


def test_hop_two_adds_payments_of_the_accounts_hop_one_met(tmp_path: Path) -> None:
    lines = [
        raw_line(10 * DAY, "S", "R"),
        raw_line(10 * DAY + 5, "S", "X"),  # hop 1 meets X
        raw_line(10 * DAY + 9, "X", "Z"),  # hop 2: X pays Z
        raw_line(10 * DAY + 20, "Z", "W"),  # three hops away: left out
    ]
    dataset = graph_dataset(tmp_path, lines)
    document = build_evidence(
        EvidenceIndex(dataset.transactions), row_of(dataset, 0), "A0001", "nearest"
    )
    hops = {t["transaction_id"]: t["hop"] for t in document["linked_transactions"]}
    assert hops == {1: 1, 2: 2}


def test_only_payments_within_seven_days_are_linked(tmp_path: Path) -> None:
    lines = [
        raw_line(10 * DAY, "S", "R"),
        raw_line(10 * DAY + 6 * DAY, "S", "A"),  # 6 days later: in
        raw_line(10 * DAY + 8 * DAY, "S", "B"),  # 8 days later: out
        raw_line(10 * DAY - 6 * DAY, "C", "R"),  # 6 days earlier: in
        raw_line(10 * DAY - 8 * DAY, "D", "R"),  # 8 days earlier: out
    ]
    dataset = graph_dataset(tmp_path, lines)
    document = build_evidence(
        EvidenceIndex(dataset.transactions), row_of(dataset, 0), "A0001", "nearest"
    )
    assert {t["transaction_id"] for t in document["linked_transactions"]} == {1, 3}


def test_the_evidence_is_capped_and_keeps_the_nearest_in_time(tmp_path: Path) -> None:
    lines = [raw_line(5 * DAY, "S", "R")]
    lines += [raw_line(5 * DAY + 1 + i, "S", "X%d" % i) for i in range(70)]
    dataset = graph_dataset(tmp_path, lines)
    document = build_evidence(
        EvidenceIndex(dataset.transactions), row_of(dataset, 0), "A0001", "nearest"
    )
    ids = [t["transaction_id"] for t in document["linked_transactions"]]
    assert len(ids) == config.EVIDENCE_MAX_EDGES == 50
    assert ids == list(range(1, 51))  # the 50 closest in time
    assert document["counts"]["truncated"] and document["counts"]["hop1_found"] == 70


def test_hop_one_fills_the_cap_before_hop_two(tmp_path: Path) -> None:
    lines = [raw_line(5 * DAY, "S", "R")]
    lines += [raw_line(5 * DAY + 1 + i, "S", "X%d" % i) for i in range(60)]
    lines += [raw_line(5 * DAY + 100, "X0", "Z")]
    dataset = graph_dataset(tmp_path, lines)
    document = build_evidence(
        EvidenceIndex(dataset.transactions), row_of(dataset, 0), "A0001", "nearest"
    )
    assert {t["hop"] for t in document["linked_transactions"]} == {1}


def test_evidence_fields_and_no_label_anywhere(tmp_path: Path) -> None:
    lines = [
        raw_line(10 * DAY, "S", "R", amount=250.5, label=1),
        raw_line(10 * DAY + 3, "S", "X", label=1),
    ]
    dataset = graph_dataset(tmp_path, lines)
    document = build_evidence(
        EvidenceIndex(dataset.transactions), 0, "A0001", "nearest"
    )
    link = document["linked_transactions"][0]
    assert set(link) == {
        "transaction_id", "hop", "time", "from_account", "to_account",
        "amount_paid", "payment_currency", "amount_received",
        "receiving_currency", "payment_format",
    }  # fmt: skip
    assert link["from_account"] == "10/S" and link["payment_format"] == "Cheque"
    assert "laundering" not in json.dumps(document).lower()


def test_evidence_is_written_as_one_json_file_per_alert(tmp_path: Path) -> None:
    dataset = graph_dataset(tmp_path, [raw_line(10 * DAY, "S", "R")])
    document = build_evidence(
        EvidenceIndex(dataset.transactions), 0, "A0007", "nearest"
    )
    path = write_evidence(document, tmp_path / "evidence")
    assert path.name == "A0007.json"
    assert json.loads(path.read_text())["alert_id"] == "A0007"


# --- "flow": follow the money ---


def flow_document(tmp_path: Path, lines, strategy: str = "flow"):
    dataset = graph_dataset(tmp_path, lines)
    index = EvidenceIndex(dataset.transactions)
    return build_evidence(index, row_of(dataset, 0), "A0001", strategy)


def flow_pairs(document):
    return {
        t["transaction_id"]: (t["hop"], t.get("flow"), t.get("flow_step"))
        for t in document["linked_transactions"]
    }


def test_flow_follows_the_receivers_onward_payments_after_the_alert(
    tmp_path: Path,
) -> None:
    lines = [
        raw_line(5 * DAY, "S", "R"),  # 0 alert
        raw_line(5 * DAY + 10, "R", "X"),  # 1 onward, step 1
        raw_line(5 * DAY + 20, "X", "Y"),  # 2 onward, step 2 (X paid after receiving)
        raw_line(5 * DAY - 10, "R", "Z"),  # 3 before the alert: not onward
    ]
    pairs = flow_pairs(flow_document(tmp_path, lines))
    assert pairs[1] == (1, None, None) or pairs[1][1] == "onward"
    assert pairs[2] == (2, "onward", 2)
    assert 3 in pairs  # R touches it, so it is a hop-1 row, but not an onward flow row
    assert pairs[3] == (1, None, None)


def test_flow_follows_the_money_back_from_the_sender_before_the_alert(
    tmp_path: Path,
) -> None:
    lines = [
        raw_line(5 * DAY, "S", "R"),  # 0 alert
        raw_line(5 * DAY - 10, "W", "S"),  # 1 backward, step 1
        raw_line(5 * DAY - 20, "V", "W"),  # 2 backward, step 2 (before W paid S)
        raw_line(5 * DAY - 5, "U", "W"),  # 3 after W paid S: wrong order, left out
    ]
    pairs = flow_pairs(flow_document(tmp_path, lines))
    assert pairs[2] == (2, "backward", 2) and 3 not in pairs


def test_a_second_step_payment_made_before_the_money_arrived_is_not_onward(
    tmp_path: Path,
) -> None:
    lines = [
        raw_line(5 * DAY, "S", "R"),
        raw_line(5 * DAY + 30, "R", "X"),  # X receives at +30
        raw_line(5 * DAY + 20, "X", "Y"),  # X pays at +20, before it received
    ]
    assert 2 not in flow_pairs(flow_document(tmp_path, lines))


def test_flow_looks_only_seven_days_ahead_and_back(tmp_path: Path) -> None:
    lines = [
        raw_line(10 * DAY, "S", "R"),
        raw_line(10 * DAY + 6 * DAY, "R", "A"),  # in
        raw_line(10 * DAY + 8 * DAY, "R", "B"),  # out
        raw_line(10 * DAY - 6 * DAY, "C", "S"),  # in
        raw_line(10 * DAY - 8 * DAY, "D", "S"),  # out
    ]
    assert set(flow_pairs(flow_document(tmp_path, lines))) == {1, 3}


def test_flow_keeps_fifteen_hop_one_rows_then_thirty_five_flow_rows(
    tmp_path: Path,
) -> None:
    lines = [raw_line(5 * DAY, "S", "R")]
    lines += [raw_line(5 * DAY + 1 + i, "R", "X%d" % i) for i in range(40)]
    lines += [raw_line(5 * DAY - 1 - i, "W%d" % i, "S") for i in range(40)]
    document = flow_document(tmp_path, lines)
    hops = [t["hop"] for t in document["linked_transactions"]]
    assert hops.count(1) == 15 and hops.count(2) == 35 and len(hops) == 50
    assert document["strategy"] == "flow" and document["counts"]["truncated"]


def test_onward_and_backward_flow_take_turns(tmp_path: Path) -> None:
    lines = [raw_line(5 * DAY, "S", "R")]
    lines += [raw_line(5 * DAY + 1 + i, "R", "X%d" % i) for i in range(60)]
    lines += [raw_line(5 * DAY - 1 - i, "W%d" % i, "S") for i in range(60)]
    document = flow_document(tmp_path, lines)
    flows = [t["flow"] for t in document["linked_transactions"] if t["hop"] == 2]
    assert abs(flows.count("onward") - flows.count("backward")) <= 1


def test_flow_with_nothing_onward_still_uses_the_backward_side(tmp_path: Path) -> None:
    lines = [raw_line(5 * DAY, "S", "R")]
    lines += [raw_line(5 * DAY - 1 - i, "W%d" % i, "S") for i in range(60)]
    flows = [
        t.get("flow") for t in flow_document(tmp_path, lines)["linked_transactions"]
    ]
    assert flows.count("backward") == 35 and flows.count("onward") == 0


def test_flow_evidence_carries_no_label(tmp_path: Path) -> None:
    lines = [
        raw_line(5 * DAY, "S", "R", label=1),
        raw_line(5 * DAY + 3, "R", "X", label=1),
    ]
    document = flow_document(tmp_path, lines)
    assert "laundering" not in json.dumps(document).lower()


def test_an_unknown_strategy_is_refused(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="Unknown evidence strategy"):
        flow_document(tmp_path, [raw_line(5 * DAY, "S", "R")], strategy="best")
