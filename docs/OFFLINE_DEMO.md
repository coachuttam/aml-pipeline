# Offline demo: the case-note student on a Mac with Ollama

The `typology2` student (Project 3) runs as a 4.7 GB Q4_K_M GGUF on a local Ollama, with no GPU and no internet. The notes go through the same fact check as before.

**Your Mac:** Apple M2 with **8 GB** of memory and 21 GB free disk.
- Memory, not disk, is the limit. The model takes about 4.7 GB and its 8,192-token context about 0.25 GB with the cache settings in step 2.
- Close the browser and other large apps while it runs.
- **Measured: about 5 minutes per note.** A0001 took 292 s for 708 output tokens (before the run, the estimate was 2-4 min). See "Speed tips" below.

**Disk:** peaks at about 9.5 GB while creating (the downloaded GGUF plus Ollama's copy). It drops to 4.7 GB after the download is deleted in step 4.

## 0. Before: the GGUF and the dev check (Colab)

1. Run `notebooks/gguf_ollama_colab.ipynb` on an L4.
2. It writes `aml-student-typology2-Q4_K_M.gguf`, its `.sha256`, and `dev_raw_Qwen2.5-7B-Instruct_typology2_q4km.jsonl` to `MyDrive/aml_finetune/`.
3. Download the dev file to `data/finetune/`, then run:

   ```bash
   python -m src.quant_check --gguf-raw data/finetune/dev_raw_Qwen2.5-7B-Instruct_typology2_q4km.jsonl
   ```

   This compares the quantised model with the adapter, **on dev only** (50 alerts), on three checks:
   - valid JSON: at most 2 fewer than the adapter;
   - invented facts per claim: at most twice the adapter's;
   - same typology: on at least 80% of alerts.

   Test stays the adapter's single scored run.

   **Result (run once): passed.**
   - Valid JSON: 50/50, the same as the adapter.
   - Invented facts: 0/1189, against the adapter's 18/1391.
   - Same typology as the adapter: 40/50 = 0.80, exactly the bar.
   - Typology right on visible patterns: 1/19, against the adapter's 3/19.
   - No note is identical to the adapter's.

   The GGUF is a close but different model. It is fine for the demo, but the reported numbers stay the adapter's.

## 1. Install Ollama

1. Download Ollama for macOS from https://ollama.com/download.
2. Move it to Applications and open it once. This installs the `ollama` command.
3. Quit it from the menu bar. Step 2 starts the server with memory-saving settings instead.

## 2. Start the server with the memory-saving settings

In a Terminal you keep open:

```bash
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
```

`OLLAMA_KV_CACHE_TYPE=q8_0` halves the context cache (it needs flash attention), which matters with 8 GB.

## 3. Download the GGUF from Drive and check it

1. In Google Drive, download `aml_finetune/aml-student-typology2-Q4_K_M.gguf` and its `.sha256`.
2. Move the GGUF into `~/aml-pipeline/models/`. That folder is git-ignored.
3. Check the download:

   ```bash
   cd ~/aml-pipeline
   shasum -a 256 models/aml-student-typology2-Q4_K_M.gguf   # must equal the .sha256 file
   ```

## 4. Create the model, then delete the download

```bash
ollama create aml-student-typology2 -f ollama/Modelfile   # copies the GGUF into ~/.ollama
ollama list                                               # aml-student-typology2, ~4.7 GB
rm models/aml-student-typology2-Q4_K_M.gguf               # the download is no longer needed
```

`ollama/Modelfile` sets:
- Qwen's chat format, exactly as in training;
- greedy decoding with `repeat_penalty 1.0` (Ollama's default would be 1.1);
- an 8,192-token context and at most 1,400 new tokens.

## 5. One alert end to end, timed

```bash
source .venv/bin/activate
python -m src.note_writer ollama --alert A0001
```

It builds the student's input from the evidence (no labels) and asks Ollama. If the answer is invalid, it retries once, as on Colab. Then it fact-checks the note and prints it with one line like `A0001: 140 s, 912 output tokens, fact check: clean`. The answer is kept in `data/finetune/ollama_demo_raw.jsonl` as demo output; it is never scored.

**Measured on the M2 (8 GB), Wi-Fi off:** `A0001: 292 s, 708 output tokens, fact check: clean`.
- The note suggests typology **RANDOM**, where the adapter's saved note says *unclear*. That is the kind of shift the dev check measured (10 of 50 typologies differ).
- The analyst confirms or ignores the suggestion as usual.

## 6. In the UI

```bash
AML_PIPELINE_NOTE_WRITER=ollama streamlit run app.py
```

The saved notes are still the adapter's. On an alert's page, **"Regenerate note (local Ollama, offline)"** writes a new one with the GGUF, fact-checked the same way. A decision made on it is logged with note version `v2-student-typology2-q4km`.

It worked with Wi-Fi off.

## Speed tips for a live demo

- **Close what you don't need:** browser tabs, mail, chat apps. With 8 GB, the model (4.7 GB) and macOS fill the memory. Any swapping makes a note much slower than 5 min.
- **Keep the memory-saving server settings** from step 2.
- **Warm it up:** run step 5 once before the audience arrives, so the model is already loaded.
- **Start the note early:** click "Regenerate note" first, then talk through the evidence table, the network graph and the goAML draft while it writes (about 5 min).
- **Have the saved note to fall back on:** the adapter's note stays on screen until you regenerate.

## Removing it

```bash
ollama rm aml-student-typology2
```

This frees the 4.7 GB in `~/.ollama`.
