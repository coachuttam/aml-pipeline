"""Score every LI-Small validation and test edge with the fine-tuned GNN (Colab GPU).

Self-contained on purpose (no imports from this repo): it is copied to Google Drive
and run in Colab next to a clone of aml-gnn-transfer, which is only read. From that
repo it reuses the data loaders, the model builder and the checkpoint loading of
`src/score_checkpoint.py`, so the model is rebuilt exactly as it was scored there.
It adds two things that scorer lacks: which transaction each score is for, and a
score for every edge of the period (the scorer skips seed edges the neighbour
sampler left out of their batch; add_missing_seed_edges puts them back).

    python gnn_export.py --gnn-repo /content/aml-gnn-transfer --output scores.parquet

Model: `pna_finetune_f1.0_s0` (Multi-PNA + EU, pre-trained on HI-Small, fine-tuned on
all LI-Small training labels). It has the best validation F1 of the fine-tuned runs.

Ids: Multi-GNN numbers edges by their row in `formatted_transactions.csv`, which is
sorted by time; that file's EdgeID column is the row of the raw CSV, the pipeline's
`transaction_id`. Port numbers are added without reordering edges, so row k of the
formatted file is edge k.

Output, one row per transaction of the period: transaction_id, score (laundering
probability), period ("validation" or "test") and added (the sampler had left the
edge out of its own batch; see add_missing_seed_edges). No label is written. As a check
that the right model was loaded, F1 on the sampled edges only (what the record
covers) is compared with the record; F1 on all edges is printed beside it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

RUN_ID = "pna_finetune_f1.0_s0"  # best validation F1 (0.332) of the fine-tuned runs
MODEL = "pna"
SEED = 0
DATASET = "LI-Small"
RECORDED = {"validation_f1": 0.33203883495145625, "test_f1": 0.3665065202470831,
            "threshold": 0.698430597782135}  # fmt: skip  # from the run record
F1_TOLERANCE = 0.01  # neighbour sampling is random, so scores move slightly per run
PERIODS = ("validation", "test")
EDGE_TYPE = ("node", "to", "node")  # Multi-GNN's forward edges (reverse MP)


def transaction_ids(positions: np.ndarray, edge_id_column: np.ndarray) -> np.ndarray:
    """Formatted-file row -> EdgeID (the raw CSV row, the pipeline's id)."""
    return edge_id_column[positions]


EXAMPLES = 5  # ids shown per kind of mismatch when the coverage check fails


def coverage_report(positions: np.ndarray, expected: np.ndarray) -> Dict:
    """Counts and a few examples: expected, scored, missing, extra, duplicates."""
    values, counts = np.unique(positions, return_counts=True)
    missing = np.setdiff1d(expected, values)
    extra = np.setdiff1d(values, expected)
    duplicates = values[counts > 1]
    report: Dict = {"expected": int(len(expected)), "scored": int(len(positions))}
    for name, ids in (("missing", missing), ("extra", extra),
                      ("duplicates", duplicates)):  # fmt: skip
        report[name] = int(len(ids))
        report[name + "_examples"] = [int(i) for i in ids[:EXAMPLES]]
    return report


def require_all_scored(positions: np.ndarray, expected: np.ndarray) -> Dict:
    """Every edge of the period scored exactly once, and no other edge."""
    report = coverage_report(positions, expected)
    print("Coverage:", report)
    if report["missing"] or report["extra"] or report["duplicates"]:
        raise ValueError("Scored edges differ from the period's edges: %s" % report)
    return report


def scores_table(periods: Dict[str, Tuple[np.ndarray, ...]], edge_ids):
    """One table: transaction_id, score, period, added (no label).

    `added` is True where the neighbour sampler left the edge out of its own batch
    and it was put back before scoring (see add_missing_seed_edges).
    """
    import pandas as pd

    parts = [
        pd.DataFrame({"transaction_id": transaction_ids(positions, edge_ids),
                      "score": scores.astype(np.float32), "period": name,
                      "added": added.astype(bool)})
        for name, (positions, scores, added) in periods.items()
    ]  # fmt: skip
    return pd.concat(parts, ignore_index=True)


def f1_check(recomputed: Dict[str, float]) -> Dict[str, str]:
    """ "matches" or "differs" for each recomputed number against the record."""
    return {
        name: "matches" if abs(value - RECORDED[name]) <= F1_TOLERANCE else "differs"
        for name, value in recomputed.items()
    }


def missing_seed_ids(seed_ids: np.ndarray, present_ids: np.ndarray) -> np.ndarray:
    """Seed edges the neighbour sampler did not put in their own batch."""
    return seed_ids[~np.isin(seed_ids, present_ids)].astype(np.int64)


def to_batch_nodes(pairs: np.ndarray, batch_node_ids: np.ndarray) -> np.ndarray:
    """Global account ids (2 x n) -> their positions among the batch's nodes."""
    local = {int(node): i for i, node in enumerate(batch_node_ids)}
    return np.vectorize(local.__getitem__, otypes=[np.int64])(pairs)


def add_missing_seed_edges(batch, data, seed_ids, reverse: bool) -> np.ndarray:
    """Put unsampled seed edges back into the batch; returns their ids.

    LinkNeighborLoader starts from the seed's two accounts and samples a capped
    number of their edges, so a busy account's own seed edge can be left out and
    is then never scored. Multi-GNN's evaluate_hetero (train_util.py) adds such
    edges back only for its Small_J / Small_Q data, and score_checkpoint.py in
    aml-gnn-transfer does not, so its recorded F1 covers sampled edges only. This
    is Multi-GNN's code, applied to LI-Small. The two accounts are always in the
    batch (they are the seeds), so only the edge is added; as there, not its
    reverse edge.
    """
    import torch

    store, full = (batch[EDGE_TYPE], data[EDGE_TYPE]) if reverse else (batch, data)
    present = store.edge_attr[:, 0].detach().cpu().numpy()
    missing = missing_seed_ids(seed_ids.numpy(), present)
    if len(missing):
        node_ids = (batch["node"].n_id if reverse else batch.n_id).cpu().numpy()
        rows = torch.from_numpy(missing)
        pairs = full.edge_index[:, rows].cpu().numpy()
        index = torch.from_numpy(to_batch_nodes(pairs, node_ids))
        store.edge_index = torch.cat((store.edge_index, index), 1)
        store.edge_attr = torch.cat((store.edge_attr, full.edge_attr[rows]), 0)
        store.y = torch.cat((store.y, full.y[rows]), 0)
    return missing


def batch_positions_and_scores(batch, loader, inds, model, device, reverse: bool):
    """Rows, labels, probabilities and "added" flags of one batch's seed edges."""
    import torch

    from src.score_checkpoint import (
        score_batch_plain,
        score_batch_reverse,
        seed_edge_mask,
    )

    forward = batch[EDGE_TYPE] if reverse else batch
    loader_attr = (loader.data[EDGE_TYPE] if reverse else loader.data).edge_attr
    inds_cpu = inds.detach().cpu()
    seed_ids = loader_attr.detach().cpu()[inds_cpu[forward.input_id.cpu()], 0]
    added = add_missing_seed_edges(batch, loader.data, seed_ids, reverse)
    mask = seed_edge_mask(forward.edge_attr, forward.input_id, loader_attr, inds)
    # Column 0 is the edge's row, added by Multi-GNN; read it before it is dropped.
    positions = forward.edge_attr[mask, 0].detach().cpu().long()
    score_batch = score_batch_reverse if reverse else score_batch_plain
    truth, scores = score_batch(batch, loader, inds, model, device)
    flags = torch.from_numpy(np.isin(positions.numpy(), added))
    return positions, truth.cpu(), scores.cpu(), flags


def collect_period(loader, inds, model, device, reverse: bool):
    """Rows, labels, probabilities and "added" flags of every seed edge."""
    import torch

    parts = []
    with torch.no_grad():
        for batch in loader:
            parts.append(
                batch_positions_and_scores(batch, loader, inds, model, device, reverse)
            )
    return [torch.cat([p[i] for p in parts]).numpy() for i in range(4)]


def load_model():
    """Data loaders and the trained model, as aml-gnn-transfer's scorer builds them.

    `src` here is aml-gnn-transfer's package (on sys.path in Colab), not this repo's.
    """
    import torch

    from src.config import DATA_ROOT
    from src.prepare import run_folder_for
    from src.records import model_folder_for_run
    from src.score_checkpoint import (
        ScoringRequest,
        checkpoint_file,
        import_multignn_modules,
        load_trained_model,
        multignn_argv,
        prepare_loaders,
        require_checkpoint,
    )  # fmt: skip

    request = ScoringRequest(RUN_ID, RUN_ID, MODEL, DATASET, SEED, True)
    os.chdir(run_folder_for(RUN_ID, DATA_ROOT))  # Multi-GNN reads its settings here
    modules = import_multignn_modules()
    args = modules.util.create_parser().parse_args(multignn_argv(request))
    modules.util.set_seed(args.seed)
    config = json.loads(Path("data_config.json").read_text())
    data = prepare_loaders(modules, args, config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path = require_checkpoint(checkpoint_file(model_folder_for_run(RUN_ID), True))
    model, epoch = load_trained_model(modules, args, data, path, device)
    print("Loaded %s (epoch %d) on %s" % (path, epoch, device))
    return data, model, device, bool(args.reverse_mp), config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gnn-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    sys.path.insert(0, str(arguments.gnn_repo))
    data, model, device, reverse, config = load_model()
    loaders = {"validation": (data.validation_loader, data.validation_indices),
               "test": (data.test_loader, data.test_indices)}  # fmt: skip
    periods, labels = {}, {}
    for name, (loader, inds) in loaders.items():
        positions, truth, scores, added = collect_period(
            loader, inds, model, device, reverse
        )
        require_all_scored(positions, inds.detach().cpu().numpy())
        print("%s: %d of %d edges added back" % (name, added.sum(), len(added)))
        periods[name], labels[name] = (positions, scores, added), truth
    report_f1(periods, labels)
    import pandas as pd

    csv = Path(config["paths"]["aml_data"]) / "Small_LI" / "formatted_transactions.csv"
    edge_ids = pd.read_csv(csv, usecols=["EdgeID"])["EdgeID"].to_numpy()
    scores_table(periods, edge_ids).to_parquet(output, index=False)
    print("Wrote %s" % output)


def recomputed_f1(periods: Dict, labels: Dict, keep: Dict) -> Dict[str, float]:
    """Validation and test F1 (threshold from validation) on the rows kept."""
    from src.metrics import score_run, score_validation

    val = (labels["validation"][keep["validation"]],
           periods["validation"][1][keep["validation"]])  # fmt: skip
    test = (labels["test"][keep["test"]], periods["test"][1][keep["test"]])
    return {"validation_f1": score_validation(*val).minority_f1,
            "test_f1": score_run(*val, *test).minority_f1}  # fmt: skip


def report_f1(periods: Dict, labels: Dict) -> None:
    """F1 like the record (sampled edges only, as score_checkpoint scored) and on all.

    The record never scored the edges added back, so only the sampled-only F1 is
    compared with it; the all-edges F1 is printed for information.
    """
    sampled = {name: ~periods[name][2] for name in PERIODS}
    everything = {name: np.ones(len(periods[name][2]), bool) for name in PERIODS}
    like_record = recomputed_f1(periods, labels, sampled)
    print("Sampled edges only:", like_record, "| record:", RECORDED)
    print("Check:", f1_check(like_record))
    print("All edges:", recomputed_f1(periods, labels, everything))


if __name__ == "__main__":
    main()
