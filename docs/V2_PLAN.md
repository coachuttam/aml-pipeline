# V2: remaining steps (plan only)

The note writer is done: the frozen `typology2` student (Project 3) (final test run once, see README). The three remaining steps, in order. Each step comes with tests and a README section, and none touches the frozen adapter or re-scores test.

## 1. GNN scores and edge colours

**Done (export run on Colab, join and UI built; see README):** `notebooks/gnn_scores_colab.ipynb` with `src/gnn_export.py` scores every LI-Small validation and test transaction with `pna_finetune_f1.0_s0` and writes `gnn_scores.parquet` (transaction_id, score, period; no label). Copy it to `artifacts/` (git-ignored).

**Join** (`src/gnn_scores.py`, CPU, with tests on made-up data):

- Read the parquet into id -> (score, period). The ids are IBM EdgeIDs, the same as `transaction_id` in the evidence, so the join is a plain lookup.
- **Coverage check before use:** every alert transaction (all in the test period) and every evidence row from 2022-09-07 on must have a score. A missing id means the id mapping is wrong, so stop.
- **Training-period rows stay unscored.** 1,663 of the 10,182 evidence rows (16%, in 116 alerts) are before 2022-09-07, in the GNN's training period. Its score there comes from a model that learned those rows' labels, so showing it would leak labels into the UI. They are drawn grey: "not scored (GNN training period)".
- For each alert: the alert transaction's GNN score, its percentile among all test-period GNN scores, and a band. "High" means at least 0.698, the best-F1 threshold chosen on validation in aml-gnn-transfer (Project 4) (`config.GNN_HIGH_THRESHOLD`); otherwise "low".

**UI:**

- **Alert detail header:** "GNN second opinion: 0.93, high (top 0.4% of test). Priority is set by the LightGBM score." The queue gets a `gnn_score` column; the default sort stays `priority_rank`.
- **Network graph:** a toggle "Colour edges by: note citations (default) | GNN score".
  - In GNN mode, edges use one sequential scale from 0 to 1 (light to dark), and edges at or above the threshold are drawn thicker. Unscored edges are grey and dashed. The alert edge keeps a red outline in both modes.
  - **Legend:** a colour bar labelled "GNN laundering score (0-1)", a tick at 0.698 marked "high (threshold from validation)", and a grey dashed sample marked "not scored: GNN training period".
  - The hover text adds "GNN 0.81".
- **Evidence table:** a `gnn_score` column, blank for unscored rows.

**Rules:**

- LightGBM keeps the priority: the GNN neither re-ranks alerts nor removes any.
- The student's input and the case note do not change. Adding a score would bring back the shortcut that broke the first student.
- Nothing is tuned on test: the threshold comes from validation, and the run was chosen by validation F1.

**Report (evaluation only, after the UI works):**

- How many of the 200 alerts the GNN calls high, among laundering alerts and among false alarms.
- Agreement with LightGBM (rank correlation).
- For laundering alerts, the share of pattern edges among the evidence edges the GNN scores highest.

These are stated with their n and are not used to choose anything.

## 2. goAML draft

- **What:** `src/goaml_draft.py` turns an alert, its analyst decision (from the audit log) and the student note into a goAML-style STR **draft** in XML. It covers the reporting entity placeholder, the transactions (cited ids only), the accounts (key_accounts) and the reason for suspicion (the note's red flags plus the analyst's comment).
- **Rules:** a draft is made only after the analyst escalates. The typology goes in only if the analyst confirmed it. It is validated against a trimmed goAML schema kept in `docs/`. Placeholders are marked, and the file is never filed automatically. Every value must trace back to the evidence (the same fact check).
- **UI:** a "Download goAML draft" button on escalated alerts.

## 3. Offline GGUF / Ollama demo

- **What:** merge the `typology2` LoRA into Qwen2.5-7B-Instruct, convert it to GGUF (llama.cpp) at Q4_K_M, and serve it with Ollama (a `Modelfile` carrying the v4 system prompt and temperature 0). Add `NOTE_WRITER = "ollama"` so the UI can regenerate a note offline on a laptop.
- **Check before use:** quantisation can change answers. Re-run **dev** (not test) with the GGUF model and compare it with the adapter's dev answers: JSON validity, typology agreement and the fact check. Test stays the single scored run of the adapter.
- **Limits to state:** a 7B Q4 model needs about 5 GB of RAM and takes seconds per note on a CPU. The model files stay out of git.
