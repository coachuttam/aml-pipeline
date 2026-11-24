# Demo script (5 minutes)

Run `streamlit run app.py` (needs `artifacts/` from steps 1 and 2; no API key needed). Two alerts: **A0029** (laundering, typology hint correct) and **A0148** (false alarm, closed). The labels are not shown in the app. They come from `artifacts/ground_truth.parquet`, which only the evaluation reads, so say which is which out loud.

| Time | Do | Say |
| --- | --- | --- |
| 0:00 | Open **Metrics** | "Synthetic IBM data. The detector is weak (F1 0.22, PR-AUC 0.12). Of the top 200 alerts, 26% are laundering. This pipeline is about explaining alerts, not detecting better." Point at the caveats list: typology is a hint, about 1 in 4 right. |
| 0:45 | **Alert queue** | "Priority is the detector score. Each row has the top reason, a typology hint and a needs-review flag." Filter *needs review only* (17 notes), then back to *all*. |
| 1:30 | **Alert detail**, pick **A0029** | Read the plain-English reasons: ACH, sender made 76 payments in 7 days, first-time pair. |
| 2:00 | Same page | Note says GATHER_SCATTER: an account receives large ACH payments and then disperses them. Highlighted rows are the cited transactions; the graph shows the hub with the alert edge in red. "Fact check passed: every id, account and amount is in the evidence." Read one investigator question. |
| 3:00 | **Decision** | Escalate, comment "hub receives then disperses, same day". Show the audit log line (time, alert, decision, note version v3). |
| 3:30 | **Alert detail**, pick **A0148** | Lower score reasons: a payment 3.3 times the receiver's usual amount, 5 payments in 7 days. Only 4 cited transactions. The note says FAN_IN, but it never said "unclear" on a false alarm. Be honest: this is the hint's weakness. |
| 4:15 | **Decision** | Close, comment "single large payment, no onward flow in the evidence". "The analyst decides; an earlier version let the LLM decide and its precision was 0.27, the base rate." |
| 4:45 | Wrap | "Every claim points at a transaction; notes that fail the check are flagged. What it does not prove: that the story is right." |

Optional: **Regenerate note** on the detail page calls OpenAI (needs `OPENAI_API_KEY`; the saved note is kept if it fails). Skip it in the demo.
