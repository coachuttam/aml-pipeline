# V2: remaining steps (plan only)

The note writer is done: the frozen `typology2` student of Project 3 (final test run once, see README). The three remaining steps, in order. Each step comes with tests and a README section, and none touches the frozen adapter or re-scores test.

## 1. GNN scores and edge colours

**Done (export run on Colab, join and UI built; see README):** `notebooks/gnn_scores_colab.ipynb` with `src/gnn_export.py` scores every LI-Small validation and test transaction with `pna_finetune_f1.0_s0` and writes `gnn_scores.parquet` (transaction_id, score, period; no label). Copy it to `artifacts/` (git-ignored).

**Join** (`src/gnn_scores.py`, CPU, with tests on made-up data):

- Read the parquet into id -> (score, period). The ids are IBM EdgeIDs, the same as `transaction_id` in the evidence, so the join is a plain lookup.
- **Coverage check before use:** every alert transaction (all in the test period) and every evidence row from 2022-09-07 on must have a score. A missing id means the id mapping is wrong, so stop.
- **Training-period rows stay unscored.** 1,663 of the 10,182 evidence rows (16%, in 116 alerts) are before 2022-09-07, in the GNN's training period. Its score there comes from a model that learned those rows' labels, so showing it would leak labels into the UI. They are drawn grey: "not scored (GNN training period)".
- For each alert: the alert transaction's GNN score, its percentile among all test-period GNN scores, and a band. "High" means at least 0.698, the best-F1 threshold chosen on validation in aml-gnn-transfer, Project 4 (`config.GNN_HIGH_THRESHOLD`); otherwise "low".

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

## 2. goAML draft (UAE first)

**Done (see README):** `src/goaml_draft.py`, `src/goaml_check.py`, `docs/goaml_style_draft.xsd`, `src/ui_goaml.py`. Drafts are made only for escalated alerts, never filed, checked against our own schema, with placeholders listed, and every draft is logged.

**Later: FIU-IND adapter (FINGate).** India's FIU takes STRs through FINGate in its own reporting format, not goAML. Plan an adapter, `src/fiu_ind_draft.py`, that maps the same inputs to that format:
- the same inputs: escalation, note, cited evidence rows and key accounts;
- INR as the local currency;
- the same placeholder and missing-fields mechanism;
- its own draft schema, labelled "not a certified FIU-IND format".

The draft builder would choose a jurisdiction (`config.STR_JURISDICTION = "UAE" | "IN"`); the decision flow and the audit log stay the same. It is not built until the FIU-IND field list has been checked against the official specification.

## 3. Offline GGUF / Ollama demo

**Built, not yet run.** The pieces:
- `notebooks/gguf_ollama_colab.ipynb`: merge the adapter, convert to GGUF, quantise to Q4_K_M; only the GGUF goes to Drive.
- `ollama/Modelfile`, `src/ollama_generate.py`, and `NOTE_WRITER="ollama"` in the UI and in `src/note_writer`.
- `src/quant_check.py`: the dev-only check, with pass rules in `config.QUANT_*`.
- `docs/OFFLINE_DEMO.md`: the Mac steps.

Still to do:
1. Run the notebook.
2. Run the dev check.
3. Time one note on the Mac (8 GB M2).
4. Record the numbers in the README.
