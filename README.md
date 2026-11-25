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

Model `gpt-4.1-mini`, temperature 0, JSON-schema output. Input = score + plain-English SHAP reasons + evidence (no label, no pattern). Totals are computed in code from the cited ids. Every cited id, account and text amount (also those written in red flags and questions) is checked against that alert's evidence; a note with any violation, or still invalid after one retry, is flagged `needs_review`. The LLM no longer decides: it gives a typology, red flags and 3 investigator questions, and alert priority is the detector score (`priority_rank`).

**Finding (v1, when the LLM still decided).** On the 200 test alerts the LLM escalated 81; 22 were laundering: precision 0.27, recall 0.43. The alert list is 26% laundering (51 of 200), so LLM escalate precision 0.27 is about the base rate and it added nothing. That is why the decision was removed. (On 30 dev alerts, v1 precision was 0.90 and v2 0.82, which shows how misleading a small dev set can be.)

Prompt v3 (v2's typology definitions, no decision, questions) was checked on the dev set first (30 top validation alerts): 30/30 valid, 0 notes with invented facts, typology 3/22 mapped. Then run once on the test set (`artifacts/eval_step2.json`):

| | v1 (200 test, 1,200-token cap, LLM decides) | v3 (200 test, 3,000-token cap, one retry) |
| --- | --- | --- |
| JSON valid | 190/200 | **200/200** |
| Typology accuracy (mapped alerts) | 13/49 = 0.27 | 13/51 = 0.25 |
| Notes with invented facts / needing review | 12/190 = 6.3% | 17/200 = 8.5% |
| Invented facts per checked claim | 0.75% | 0.27% |
| Cost | $0.67 | $0.72 (about 6.5 s per note) |

Typology accuracy is poor: about one in four, so the typology is a hint for the analyst, not a result. The fact check is what makes the notes usable: 183 of 200 notes had every cited id, account and amount in the evidence, and the other 17 are flagged for review. Prompts v1 and v2 are kept for the record but target the old schema (with a decision).

## V2 step 1: fine-tuning data for the case-note model

A teacher (`gpt-4.1-mini`, temperature 0) writes gold case notes for HI-Small alerts. It sees the student's exact input plus a `PRIVILEGED_TRUTH` block (typology and pattern transaction ids, or "not laundering"); the student never does (documented exception in AGENTS.md). Alerts are the top-scored HI-Small validation and test transactions from the same LightGBM; LI-Small test alerts are never used for training. A teacher note is kept only if it passes the fact check, uses the typology it was given and cites the pattern transactions.

- **2,200 candidates, 2,159 kept** (41 dropped, almost all for a transaction id or amount not in the evidence). 1,154 of the kept are laundering; of the 2,200 candidates 54% were laundering, below the 60% aimed for, because the cap of 8 alerts per pattern left too few. Teacher cost $7.74.
- **Train 1,248 / validation 219 / dev 50 (LI-Small validation period) / test 200 (the existing alerts)**. Train and validation share no pattern. Train was rebalanced: all 749 typed examples kept, "unclear" thinned to 40% (382 false alarms and 117 laundering with thin evidence, seeded, same proportions as before). Validation is unchanged (60% unclear).
- **Student input** is the V1 input written without indentation (the same content): with Qwen's one-token-per-digit numbers the indented form needs 8.8k tokens on average and does not fit 8,192; compact it is 7.1k on average, 8,046 at most.
- **`pattern_visible`** (at least 3 of the pattern's transactions in the evidence) is stored per example in `data/finetune/meta.jsonl`. Where it is false the teacher was told to answer "unclear".

### Evaluation scheme (V1, prompt v3 and the student alike)

`src/eval_v2.py`. Five measures, on the 200 test alerts (37 laundering alerts with a visible pattern, 147 false alarms):

| | V1 | v3 |
| --- | --- | --- |
| Typology accuracy, visible-pattern laundering | 12/35 = 34% | 12/37 = 32% |
| "Unclear" rate on false alarms | 0/139 | 0/147 |
| "Unclear" rate on visible-pattern laundering | 0/35 | 0/37 |
| Invented facts per claim | 53/7,033 = 0.75% | 20/7,403 = 0.27% |
| JSON validity | 190/200 | 200/200 |

Neither prompt ever answers "unclear", even on false alarms. That, and typology on visible patterns, is what the student is meant to improve. The student is not trained or scored yet.

### Student training (not run)

`notebooks/finetune_colab.ipynb`: QLoRA on `Qwen/Qwen2.5-7B-Instruct` (set in one config cell; the notebook prints the model card's licence, listed as Apache-2.0), 8,192 tokens, 2 epochs, validation loss each epoch, adapter saved to Drive. `src/student_generate.py` (run in Colab) writes the notes for the 50 dev alerts, then the 200 test alerts once; `python -m src.score_student --set dev|test --raw <file>` applies the same fact check and `eval_v2` on a CPU. Estimate, not measured: about 9-10 h on an L4 for training (7-11 h range) and about 1 h for the 250 notes, roughly 20 Colab compute units (15-25). Run with `MAX_STEPS = 3` first to measure the real speed. Memory at 8k tokens with a 152k vocabulary on 24 GB is untested.

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

## Step 3: review UI

```bash
source .venv/bin/activate && pip install -r requirements.txt
streamlit run app.py        # reads artifacts/ only; no API key needed
```

Pages: alert queue (filter, sort), alert detail (reasons, evidence with cited rows highlighted, network graph, case note, fact-check status), a decision form at the bottom of the detail page (escalate or close with a comment, saved to `artifacts/audit_log.jsonl`, git-ignored; the Audit log page lists it) and metrics with caveats. The optional "Regenerate note" button is the only live OpenAI call. A 5-minute script is in `docs/DEMO.md`.

## Limits

- ACH is a known shortcut in IBM's synthetic data: the payment format is often the strongest SHAP reason, so do not read a high score as real-world suspicion.
- Case-note totals are per currency (never summed across currencies).

- Step 2 needs `OPENAI_API_KEY` in `.env`; `python -m src.run_dev` then `python -m src.run_step2 --set dev|test --prompt v1|v2`.

- Synthetic data; the detector is weak (about 0.21 test F1), so many alerts are false positives.
- Reason codes come from a model with a saturated score (many alerts score 1.0), so their ranking among the top alerts is coarse.
- Evidence is cut to 50 transactions, so it finds only about half of a laundering pattern's transactions on average (0.49), whichever of the two strategies is used.
- **BIPARTITE and STACK have no typed training examples.** Their patterns are large, so the 50-transaction evidence shows fewer than 3 of their transactions for 89 of 89 BIPARTITE and 145 of 150 STACK candidates; the teacher was told "unclear" and the student cannot learn these two typologies from this set. The test set has the same limit (evidence shows about half of a pattern on average).
- "Unclear" is 40% of the rebalanced train split but 60% of validation, so validation loss is not comparable to the train mix.
- Teacher notes are written with knowledge of the truth. A note can be right about the typology for reasons the student cannot see in the evidence, so student accuracy may stay below the teacher's.
