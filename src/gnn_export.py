"""Score every LI-Small validation and test edge with the fine-tuned GNN (Colab GPU).

Self-contained on purpose (no imports from this repo): it is copied to Google Drive
and run in Colab next to a clone of aml-gnn-transfer, which is only read. From that
repo it reuses the data loaders, the model builder and the checkpoint loading of
`src/score_checkpoint.py`, so the model is rebuilt exactly as it was scored there.
It adds the one thing that scorer does not keep: which transaction each score is
for.

    python gnn_export.py --gnn-repo /content/aml-gnn-transfer --output scores.parquet

Model: `pna_finetune_f1.0_s0` (Multi-PNA + EU, pre-trained on HI-Small, fine-tuned on
all LI-Small training labels). It has the best validation F1 of the fine-tuned runs.

Ids: Multi-GNN numbers edges by their row in `formatted_transactions.csv`, which is
sorted by time; that file's EdgeID column is the row of the raw CSV, the pipeline's
`transaction_id`. Port numbers are added without reordering edges, so row k of the
formatted file is edge k.

Output, one row per scored transaction: transaction_id, score (laundering
probability) and period ("validation" or "test"). No label is written. As a check
that the right model was loaded, validation and test F1 are recomputed from the
scores and printed beside the recorded ones.
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


def transaction_ids(positions: np.ndarray, edge_id_column: np.ndarray) -> np.ndarray:
    """Formatted-file row -> EdgeID (the raw CSV row, the pipeline's id)."""
    return edge_id_column[positions]


def require_all_scored(positions: np.ndarray, expected: np.ndarray) -> None:
    """Every edge of the period scored exactly once, and no other edge."""
    if len(np.unique(positions)) != len(positions):
        raise ValueError("An edge was scored twice")
    if not np.array_equal(np.sort(positions), np.sort(expected)):
        raise ValueError("Scored edges differ from the period's edges")


def scores_table(periods: Dict[str, Tuple[np.ndarray, np.ndarray]], edge_ids):
    """One table: transaction_id, score, period (no label)."""
    import pandas as pd

    parts = [
        pd.DataFrame({"transaction_id": transaction_ids(positions, edge_ids),
                      "score": scores.astype(np.float32), "period": name})
        for name, (positions, scores) in periods.items()
    ]  # fmt: skip
    return pd.concat(parts, ignore_index=True)


def f1_check(recomputed: Dict[str, float]) -> Dict[str, str]:
    """ "matches" or "differs" for each recomputed number against the record."""
    return {
        name: "matches" if abs(value - RECORDED[name]) <= F1_TOLERANCE else "differs"
        for name, value in recomputed.items()
    }


def batch_positions_and_scores(batch, loader, inds, model, device, reverse: bool):
    """Formatted-file rows, labels and probabilities of one batch's seed edges."""
    from src.score_checkpoint import (
        EDGE_TYPE,
        score_batch_plain,
        score_batch_reverse,
        seed_edge_mask,
    )

    forward = batch[EDGE_TYPE] if reverse else batch
    loader_attr = (loader.data[EDGE_TYPE] if reverse else loader.data).edge_attr
    mask = seed_edge_mask(forward.edge_attr, forward.input_id, loader_attr, inds)
    # Column 0 is the edge's row, added by Multi-GNN; read it before it is dropped.
    positions = forward.edge_attr[mask, 0].detach().cpu().long()
    score_batch = score_batch_reverse if reverse else score_batch_plain
    truth, scores = score_batch(batch, loader, inds, model, device)
    return positions, truth.cpu(), scores.cpu()


def collect_period(loader, inds, model, device, reverse: bool):
    """Rows, labels and probabilities of every edge one loader scores."""
    import torch

    parts = []
    with torch.no_grad():
        for batch in loader:
            parts.append(
                batch_positions_and_scores(batch, loader, inds, model, device, reverse)
            )
    return [torch.cat([p[i] for p in parts]).numpy() for i in range(3)]


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
        positions, truth, scores = collect_period(loader, inds, model, device, reverse)
        require_all_scored(positions, inds.detach().cpu().numpy())
        periods[name], labels[name] = (positions, scores), truth
    report_f1(periods, labels)
    import pandas as pd

    csv = Path(config["paths"]["aml_data"]) / "Small_LI" / "formatted_transactions.csv"
    edge_ids = pd.read_csv(csv, usecols=["EdgeID"])["EdgeID"].to_numpy()
    scores_table(periods, edge_ids).to_parquet(output, index=False)
    print("Wrote %s" % output)


def report_f1(periods: Dict, labels: Dict) -> None:
    """Recompute validation and test F1 (threshold from validation) and compare."""
    from src.metrics import score_run, score_validation

    validation = (labels["validation"], periods["validation"][1])
    test = (labels["test"], periods["test"][1])
    recomputed = {
        "validation_f1": score_validation(*validation).minority_f1,
        "test_f1": score_run(*validation, *test).minority_f1,
    }
    print("Recomputed:", recomputed, "| record:", RECORDED)
    print("Check:", f1_check(recomputed))


if __name__ == "__main__":
    main()
