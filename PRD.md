# PRD: Explainable AML pipeline (V1)

Status: step 1 built (detect, evidence, ground truth); LLM note and UI planned · Owner: Uttam

## 1. Problem

A model that ranks transactions is not enough for an AML team. An analyst must be able to see why a transaction was flagged, look at the surrounding activity, and write a case note that a reviewer can check. Most tools stop at the score.

## 2. Goal

One pipeline on IBM's synthetic LI-Small data: **detect -> explain -> evidence -> LLM case note -> UI**. Each step uses the output of the one before and nothing else, so every statement in a note can be traced to a transaction.

## 3. Users

| User | Need |
| --- | --- |
| AML analyst (the user the UI imitates) | Top alerts with plain reasons, linked transactions, a draft case note to edit |
| Reviewer or hiring manager | Evidence that the owner can build an explainable pipeline end to end and test it honestly |
| Owner | A portfolio project that joins Project 1 (LLM case notes), Project 2 (LightGBM, features), Project 3 (fine-tuned case-note student) and Project 4 (GNN transfer, aml-gnn-transfer) |

## 4. Steps and requirements

| Step | Output | Status |
| --- | --- | --- |
| 1a Detect | LightGBM (the aml-gnn-transfer Phase E final config) scores the LI-Small test period; top 200 alerts with score and SHAP top-5 reason codes -> `artifacts/alerts.parquet` | built |
| 1b Evidence | Per alert, the linked transactions (2 hops, +/- 7 days, at most 50; "nearest" strategy, compared with a "follow the money" one by pattern coverage): id, accounts, amount, currency, format, time -> `artifacts/evidence/<alert_id>.json` | built |
| 1c Ground truth | `is_laundering` and, if `LI-Small_Patterns.txt` is available, IBM pattern type -> `artifacts/ground_truth.parquet`, evaluation only | built |
| 2 LLM case note | Draft note per alert from alert + reasons + evidence only, every claim cited to a transaction id | built (prompt only; no LLM decision, 3 investigator questions, `needs_review` flag) |
| 3 Evaluation | Notes scored against ground truth (is the flagged pattern named correctly, are citations real, are claims supported) | planned |
| 4 UI | Alert list, reason codes, evidence table, case note | built (Streamlit, `app.py`) |

## 5. Non-goals (V1)

- Real bank data. Everything is IBM's synthetic data.
- A better detector. The Phase E LightGBM is a simple baseline; improving it is out of scope here.
- Production claims. This is a research prototype.

## 6. Success criteria for step 1

- The detector retrained here reproduces the Phase E record's test scores.
- 200 alerts, each with five reason codes and an evidence file.
- No label reaches the alert table or the evidence files (tested).
- Tests pass; no data in git.

## 7. Risks

| Risk | Handling |
| --- | --- |
| The detector is weak (Phase E test F1 about 0.21 at 100% labels) | State it; the pipeline is about explanation and evidence, not a strong detector |
| `LI-Small_Patterns.txt` is missing | Pattern types stay empty; the label is still kept; the summary says so |
| An LLM invents facts | Step 2 restricts input to the evidence file and checks every cited id |
