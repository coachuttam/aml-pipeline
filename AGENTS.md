# AGENTS.md

Instructions for AI coding agents (Claude Code, Cursor, Codex) working in this repo. Read PRD.md for what the pipeline must do and ARCHITECTURE.md for how it fits together.

## Setup and checks

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q          # must stay green
ruff format . && ruff check .
python -m src.run_step1      # detect -> evidence -> ground truth, needs the LI-Small CSV
```

LightGBM on a Mac without Homebrew needs an OpenMP library: see README.

## Hard rules

1. **Never tune on the test period.** Class weight, trees and the alert threshold are chosen on validation only. Test is scored, never used to choose.
2. **Ground truth is for evaluation only.** `is_laundering` and IBM pattern types are written to `artifacts/ground_truth.parquet` and read only by evaluation code. The alert table, the evidence files and everything the LLM sees must contain no label and no pattern type. Tests check this. **One exception, V2 training data only:** the teacher that writes gold case notes for fine-tuning (`src/teacher.py`) gets a `PRIVILEGED_TRUTH` block (label, pattern typology, pattern transaction ids) from HI-Small ground truth. This is training data, never a decision. The student's input (system and user message) must still hold no truth, a teacher note is dropped if it mentions the truth or fails the fact check, and LI-Small test alerts are never used for training. Tests check this.
3. **Do not modify other repos.** `~/aml-gnn-transfer` (Project 4) and Projects 1 and 2 are read or copied from, never edited. Say in the file header where copied code came from.
4. **No data, models or secrets in git.** Raw CSVs, the feature cache, trained models, `artifacts/*` and `kaggle.json` stay out. Only code, docs and tests are committed.
5. **Tests for each module.** A new module comes with a test file in the same change. Tests use small made-up data, never the real CSV.
6. **Explain from evidence only (from step 2 on).** An LLM case note may use only the alert, its reason codes and its evidence file. Every claim must point to a transaction id in the evidence. If the evidence is thin, the note says so.
7. **Report honestly.** If the detector is weak or a check fails, the README says so with the numbers.

## Readable code

The owner will read, run and explain every file.

- Plain, descriptive names. No abbreviations a newcomer would have to guess.
- One job per function, about 20 lines or fewer. Files under about 300 lines.
- A short docstring on every module and public function, with a tiny example where it helps.
- Comments explain why, not what.
- No magic numbers: named constants in `src/config.py`, each with a one-line reason.
- Type hints. `ruff format` and `ruff check` clean.
- Test names read as sentences.

## Conventions

- Python 3.9 compatible: start files with `from __future__ import annotations`, use `typing.List` and `Optional`.
- Alert ids are `A0001`, `A0002`, ... by rank. Transaction ids are IBM's EdgeID (the row of the raw CSV).
- Metric definitions match Project 2 and aml-gnn-transfer (Project 4).
- Keep Claude Code prompts short.

## Definition of done

Tests pass, `ruff` is clean, no data in git, the other repos have no changes, README updated with numbers and limits.
