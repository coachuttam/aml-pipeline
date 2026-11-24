# AML pipeline

Explainable AML pipeline V1 on IBM's synthetic LI-Small data: **detect -> explain -> evidence -> LLM case note -> UI**. Step 1 (detect, reason codes, evidence, ground truth) is built. The LLM note and the UI are not.

Docs: [PRD](PRD.md) · [Architecture](ARCHITECTURE.md) · [Agent rules](AGENTS.md)

## Step 1 result

The detector is the final LightGBM from [aml-gnn-transfer](../aml-gnn-transfer) (Phase E, a simple baseline). Retrained here, it reproduces that record exactly (test minority F1 0.2155, PR-AUC 0.1175).

- **200 alerts**, the highest-scored test transactions, each with a score and five SHAP reason codes: `artifacts/alerts.parquet`.
- **53 of the 200 (26.5%) are laundering** (precision@100 over the whole test period is 0.44). The validation threshold flags 479 test transactions in all; the top 200 are all above it.
- **Evidence per alert:** `artifacts/evidence/<alert_id>.json`, linked transactions within 7 days and 2 hops, at most 50 (the nearest in time; each file says how many existed). Two strategies were compared by pattern coverage (below) and the simpler "nearest" one is kept.
- **Ground truth** (evaluation only, never for the LLM): `artifacts/ground_truth.parquet` with `is_laundering` and IBM pattern type. **51 of the 53 laundering alerts map to a known pattern** (scatter-gather 11, gather-scatter 11, fan-out 8, random 6, bipartite 5, cycle 4, stack 3, fan-in 3); 2 are not in `LI-Small_Patterns.txt`.

## Evidence strategy: pattern coverage

Pattern coverage is the share of the true pattern's transactions found in an alert's evidence (the alert transaction counts), averaged over the 51 laundering alerts with a known pattern (mean pattern size 16.3). Evaluation only; it uses the patterns file.

| Strategy | Mean coverage | Median | Alerts with full coverage |
| --- | --- | --- | --- |
| nearest: 50 nearest in time within 2 hops | **0.491** | 0.50 | 7 |
| flow: 15 nearest at hop 1, then 35 along the money trail (onward from the receiver after the alert, backward from the sender before it, 7 days) | 0.486 | 0.50 | 3 |

The two are the same within noise, so "following the money" did not help here, and the higher one ("nearest") is kept (`config.EVIDENCE_STRATEGY`). The flow evidence is still written to `artifacts/evidence_flow/` for comparison. Caveat: this choice was made on the 200 test alerts' ground truth, between two options only; it is not a tuned result. The pipeline's own steps (detector, reasons, evidence) never read a label.

## Step 2 result: prompt-only LLM case notes

Model `gpt-4.1-mini`, temperature 0, JSON-schema output, input = score + plain-English SHAP reasons + evidence (no label, no pattern). Totals are computed in code from the cited ids; every cited id, account and text amount is checked against that alert's evidence.

Prompt choice on a dev set (top 30 validation-period alerts, 21 laundering; the test period was not used to choose):

| Prompt | JSON valid | Escalate precision | Recall | Typology acc. (21) | Notes with invented facts | Cost |
| --- | --- | --- | --- | --- | --- | --- |
| v1 | 29/30 | 0.90 | 0.43 | 0.10 | 0 | $0.10 |
| v2 (AMLSim definitions) | 29/30 | 0.82 | 0.43 | 0.10 | 1 | $0.10 |

v1 kept: v2's typology definitions did not help. Test run of v1 on all 200 alerts (`artifacts/eval_step2.json`):

- JSON valid 190/200 (the 10 failures hit the 1,200-token output limit on long notes; they count as failures, not retried).
- Escalate: 81 escalated, 22 truly laundering: precision 0.27, recall 0.43 (51 laundering alerts). Alert base rate is 0.26, so escalation precision is no better than the alert list itself.
- Typology accuracy 13/49 = 0.27 (valid notes among the 51 mapped alerts).
- Invented facts: 12/190 notes (6.3%) cite an id, account or amount not in the evidence; 0.75% of checked claims.
- Cost $0.67 for 200 notes, about 5.2 s per note.

The notes are readable and cited, but the LLM is a weak judge of these alerts: it does not separate real from false alerts and rarely names the IBM pattern. Totals computed by the model itself were wrong in all 10 probe notes, which is why code now computes them.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m src.run_step1      # about 10 minutes the first time (features are cached)
```

The raw CSV is read in place from `~/aml-gnn-transfer/data/Small_LI/` (set `AML_PIPELINE_RAW_CSV` to use another path). Nothing from `artifacts/`, `cache/` or the data is committed.

LightGBM on a Mac without Homebrew fails with "Library not loaded: libomp.dylib". Either `brew install libomp`, or point the library at another copy, for example Anaconda's:

```bash
L=.venv/lib/python3.9/site-packages/lightgbm/lib/lib_lightgbm.dylib
install_name_tool -add_rpath "$HOME/anaconda3/lib" "$L" && codesign --force -s - "$L"
```

## Limits

- Step 2 needs `OPENAI_API_KEY` in `.env`; `python -m src.run_dev` then `python -m src.run_step2 --set dev|test --prompt v1|v2`.

- Synthetic data; the detector is weak (about 0.21 test F1), so many alerts are false positives.
- Reason codes come from a model with a saturated score (many alerts score 1.0), so their ranking among the top alerts is coarse.
- Evidence is cut to 50 transactions, so it finds only about half of a laundering pattern's transactions on average (0.49), whichever of the two strategies is used.
