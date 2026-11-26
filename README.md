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

- **2,400 candidates, 2,399 kept** (1 dropped: it mentioned a "label"). 1,237 of the kept are laundering. 2,145 notes were reused from the earlier dataset and 255 regenerated (47 flagged as relying on reason sentences, 8 earlier drops, 200 new false alarms): teacher cost $0.86 this round (the first dataset cost $7.74, the second $0.55).
- **Train 2,180 / validation 219 / dev 50 (LI-Small validation period) / test 200 (the existing alerts)**. Train and validation share no pattern. Train keeps every laundering example (796 typed, 308 laundering whose evidence is too thin, answered "unclear") and 1,076 false alarms: **50.6% laundering** (the target was 50%; 28 more false alarms would have been needed for exactly 50%, and 200 train-only false alarms were added from the pool for this). So **"unclear" is 63.5% of train** (false alarms 49% plus thin laundering 14%) and 53% of validation, which is unchanged. The test alerts are 26.5% laundering.
- **Student input (prompt v4) is the evidence only**: the alert transaction id and the transactions. No model score, no SHAP reasons, no alert id and no count of linked transactions (that count is 210 / 410 in the median for false alarms / laundering in train but 687 / 2,058 on dev, so a student could learn thresholds that do not carry over). The reasons stay in the UI and the alert table. It is written without indentation because with Qwen's one-token-per-digit numbers the indented form does not fit 8,192; an example is 6.9k tokens on average, 8,075 at most. Prompt v4 is v3 without the sentences that mention the score and the reasons; the teacher prompt likewise.
- **Reused notes** were checked for claims that only the removed input could support (counts over the last hour or days, new senders, usual amounts, "times the", earlier payments, the found count): 47 of 2,192 were regenerated. This is a wording check, so a subtler dependence on the reasons the teacher once saw cannot be excluded; regenerating all notes would cost about $8.
- **`pattern_visible`** (at least 3 of the pattern's transactions in the evidence) is stored per example in `data/finetune/meta.jsonl`. Where it is false the teacher was told to answer "unclear".

### First student, and what was wrong

The first student (trained on the scored input) scored 10/19 typology on the 19 dev alerts with a visible pattern, below always answering GATHER_SCATTER (11/19); it answered "unclear" for none of the 20 valid false alarms and GATHER_SCATTER for 42 of 50 answers. The input format was identical to training. The cause was a shortcut: in training an alert with score 1.0 was laundering 96% of the time (35 saturated false alarms exist in the whole HI-Small pool of 60,000), while all 50 dev alerts score 1.0 and 42% are false alarms; the first reason also differed ("payment format is ACH" 93% in train, 26% in dev). The score, the reasons and the found count are therefore gone from the input.

Cue distributions are in `data/finetune/cue_report.json` (`python -m src.run_finetune cues`); they describe the alerts, not what the student sees any more. They could not be matched by choosing different HI-Small alerts:

| | Train, first | Train, now | Dev | Test (reported, not used) |
| --- | --- | --- | --- | --- |
| Laundering share | 69% | 51% | 58% | 26.5% |
| Score saturated at 1.0 | 32% | 23% | 100% | 82% |
| Laundering share inside score 1.0 | 96% | 93% | 58% | 33% |
| First reason is payment format | 93% | 89% | 26% | 35% |
| First reason is sender_sent_7d | 0% | 3% | 74% | 43% |

Prompt v4 (OpenAI) on the new evidence-only dev inputs, cost $0.19: typology 2/18 on visible-pattern laundering (it answers FAN_OUT for 36 of 49 notes), "unclear" 0/21 on false alarms, invented facts 2/1,783, JSON 49/50. With the score and reasons it scored 8/19 (v3: 5/19), so the reasons were doing the work for the LLM; always answering GATHER_SCATTER gets 11/19. The retrained student is not trained or scored yet.

### Evaluation scheme (V1, prompt v3 and the student alike)

`src/eval_v2.py`. Five measures, on the 200 test alerts (37 laundering alerts with a visible pattern, 147 false alarms):

| | V1 | v3 |
| --- | --- | --- |
| Typology accuracy, visible-pattern laundering | 12/35 = 34% | 12/37 = 32% |
| "Unclear" rate on false alarms | 0/139 | 0/147 |
| "Unclear" rate on visible-pattern laundering | 0/35 | 0/37 |
| Invented facts per claim | 53/7,033 = 0.75% | 20/7,403 = 0.27% |
| JSON validity | 190/200 | 200/200 |

Neither prompt ever answers "unclear", even on false alarms. That, and typology on visible patterns, is what the student is meant to improve. These are the scored-input prompts; prompt v4 has not been run on the test set. The retrained student is not trained or scored yet.

### Student training (not run)

`notebooks/finetune_colab.ipynb`: QLoRA on `Qwen/Qwen2.5-7B-Instruct` (set in one config cell; the notebook prints the model card's licence, listed as Apache-2.0), 8,192 tokens, 1 epoch, validation loss at the end, adapter saved to Drive. `src/student_generate.py` (run in Colab) writes the notes for the 50 dev alerts, then the 200 test alerts once (at most 1,400 new tokens; an answer cut off there is invalid; one retry with repetition penalty 1.1 for an invalid answer); `python -m src.score_student --set dev|test --raw <file>` (the notebook's answer files are named `dev_raw_<model>_<run>.jsonl`, so an old run's answers are never reused) applies the same fact check and `eval_v2` on a CPU. Drive folder `aml_finetune/` needs exactly: `train.jsonl`, `validation.jsonl`, `dev_inputs.jsonl`, `test_inputs.jsonl` (from `data/finetune/`) and `student_generate.py` (from `src/`); re-upload all five for each data version (the run name `evidence1` keeps adapters apart). Estimate, not measured: about 7.5-8 h on an L4 for the one training epoch (6-9.5 h range) and about 1 h for the 250 notes, roughly 15 Colab compute units (11-18). Run with `MAX_STEPS = 3` first to measure the real speed. The first smoke test ran out of memory in the loss (8k x 152k logits), so the notebook now uses Liger's fused cross-entropy (the logits are never built), `expandable_segments`, pinned package versions and a `USE_SMALLER_MODEL` switch (Qwen2.5-3B-Instruct); these fixes are not yet confirmed on a GPU.

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
- **BIPARTITE and STACK have no typed training examples.** Their patterns are large, so the 50-transaction evidence shows fewer than 3 of their transactions for nearly all BIPARTITE and STACK candidates; the teacher was told "unclear" and the student cannot learn these two typologies from this set. The test set has the same limit (evidence shows about half of a pattern on average).
- "Unclear" is 40% of the rebalanced train split but 60% of validation, so validation loss is not comparable to the train mix.
- Teacher notes are written with knowledge of the truth. A note can be right about the typology for reasons the student cannot see in the evidence, so student accuracy may stay below the teacher's.
- Train is 50.6% laundering (test alerts: 26.5%, dev: 58%) and 63.5% "unclear"; a student could say "unclear" too often on real laundering with a visible pattern, which the "unclear on visible laundering" measure shows.
