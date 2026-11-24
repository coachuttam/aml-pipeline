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
        EvidenceIndex(dataset.transactions), row_of(dataset, 0), "A0001"
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
        EvidenceIndex(dataset.transactions), row_of(dataset, 0), "A0001"
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
        EvidenceIndex(dataset.transactions), row_of(dataset, 0), "A0001"
    )
    assert {t["transaction_id"] for t in document["linked_transactions"]} == {1, 3}


def test_the_evidence_is_capped_and_keeps_the_nearest_in_time(tmp_path: Path) -> None:
    lines = [raw_line(5 * DAY, "S", "R")]
    lines += [raw_line(5 * DAY + 1 + i, "S", "X%d" % i) for i in range(70)]
    dataset = graph_dataset(tmp_path, lines)
    document = build_evidence(
        EvidenceIndex(dataset.transactions), row_of(dataset, 0), "A0001"
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
        EvidenceIndex(dataset.transactions), row_of(dataset, 0), "A0001"
    )
    assert {t["hop"] for t in document["linked_transactions"]} == {1}


def test_evidence_fields_and_no_label_anywhere(tmp_path: Path) -> None:
    lines = [
        raw_line(10 * DAY, "S", "R", amount=250.5, label=1),
        raw_line(10 * DAY + 3, "S", "X", label=1),
    ]
    dataset = graph_dataset(tmp_path, lines)
    document = build_evidence(EvidenceIndex(dataset.transactions), 0, "A0001")
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
    document = build_evidence(EvidenceIndex(dataset.transactions), 0, "A0007")
    path = write_evidence(document, tmp_path / "evidence")
    assert path.name == "A0007.json"
    assert json.loads(path.read_text())["alert_id"] == "A0007"
