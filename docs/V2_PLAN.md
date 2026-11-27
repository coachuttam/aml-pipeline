# V2: remaining steps (plan only)

The note writer is done: the frozen `typology2` student (final test run once, see README). The three remaining steps, in order. Each step comes with tests and a README section, and none touches the frozen adapter or re-scores test.

## 1. GNN scores and edge colours

- **What:** the LI-Small fine-tuned Multi-PNA + EU from `~/aml-gnn-transfer` (Phase D, 100% labels) scores every edge in each alert's evidence. Its checkpoint is read, never edited. The score is shown next to the LightGBM score, and the network graph colours each edge by GNN score.
- **How:** run a one-off export on Colab with the existing checkpoint (`gnn_scores.parquet`: transaction_id → score, test period only). Then `src/gnn_scores.py` joins it to the evidence, and `ui_graph.py` colours the edges on a sequential scale. The alert edge stays red and the cited edges keep their outline.
- **Rules:** LightGBM keeps the priority. The GNN score is a second opinion, not a re-ranking. The threshold for "high" comes from GNN validation, not test. The student's input does not change (adding the score would bring back the shortcut from the first student).
- **Measure:** for each alert, the share of pattern edges among the top-GNN-scored evidence edges (evaluation only). It is reported, not tuned.

## 2. goAML draft

- **What:** `src/goaml_draft.py` turns an alert, its analyst decision (from the audit log) and the student note into a goAML-style STR **draft** in XML. It covers the reporting entity placeholder, the transactions (cited ids only), the accounts (key_accounts) and the reason for suspicion (the note's red flags plus the analyst's comment).
- **Rules:** a draft is made only after the analyst escalates. The typology goes in only if the analyst confirmed it. It is validated against a trimmed goAML schema kept in `docs/`. Placeholders are marked, and the file is never filed automatically. Every value must trace back to the evidence (the same fact check).
- **UI:** a "Download goAML draft" button on escalated alerts.

## 3. Offline GGUF / Ollama demo

- **What:** merge the `typology2` LoRA into Qwen2.5-7B-Instruct, convert it to GGUF (llama.cpp) at Q4_K_M, and serve it with Ollama (a `Modelfile` carrying the v4 system prompt and temperature 0). Add `NOTE_WRITER = "ollama"` so the UI can regenerate a note offline on a laptop.
- **Check before use:** quantisation can change answers. Re-run **dev** (not test) with the GGUF model and compare it with the adapter's dev answers: JSON validity, typology agreement and the fact check. Test stays the single scored run of the adapter.
- **Limits to state:** a 7B Q4 model needs about 5 GB of RAM and takes seconds per note on a CPU. The model files stay out of git.
