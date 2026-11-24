# Architecture: Explainable AML pipeline

Read PRD.md for what the pipeline must do and AGENTS.md for the rules.

## Flow (step 1)

```
LI-Small_Trans.csv  (read in place from ~/aml-gnn-transfer/data/Small_LI)
        │
        ▼
src/data.py ─────────► Dataset: transactions (no label), truth (labels), split (60/20/20 by day)
        │
        ├─► src/features.py ──► 29 past-only features (cached in cache/, not in git)
        │         (features_base.py + features_extra.py, copied from aml-gnn-transfer)
        ▼
src/model.py ────────► LightGBM trained on the training period; class weight and
        │              threshold chosen on validation
        ├─► src/reproduce.py ► compares test scores with the Phase E record (read only)
        ▼
src/detect.py ───────► top 200 test alerts + SHAP top-5 reasons ─► artifacts/alerts.parquet
        │
        ▼
src/evidence.py ─────► linked transactions per alert, from the label-free table
        │               ─► artifacts/evidence/<alert_id>.json
        ▼
src/ground_truth.py ─► is_laundering (+ pattern type) ─► artifacts/ground_truth.parquet
                         (evaluation only; the LLM never reads it)
src/run_step1.py      runs all of the above and writes artifacts/step1_summary.json
```

## Where labels may go

Labels exist in one array (`Dataset.truth`). They are read to train the detector (training rows), to choose the threshold (validation rows), to score it, and to write the ground-truth file. The transactions table, the alert table and the evidence files are built from the label-free table, so they cannot carry a label.

## Folders

```
aml-pipeline/
├─ README.md  PRD.md  ARCHITECTURE.md  AGENTS.md
├─ src/        config, data, features(_base, _extra), model, metrics, detect,
│              evidence, ground_truth, reproduce, run_step1
├─ tests/      one test file per module, small made-up data
├─ artifacts/  generated files (git-ignored)
└─ cache/      feature matrix (git-ignored)
```

## Detector settings

The Phase E final configuration of aml-gnn-transfer: all training labels (seed 0), up to 2000 trees, early stopping on validation PR-AUC with patience 100, learning rate 0.05, 7 leaves, class weight from (0.5, 1, 2, 5) by validation PR-AUC, threshold at the best validation F1.

## Evidence rules

For an alert (S pays R at time T): hop 1 is other transactions touching S or R within +/- 7 days of T; hop 2 is transactions touching the accounts that hop 1 met. At most 50 are kept: hop 1 first, nearest in time first. Each file records how many existed and whether it was cut.
