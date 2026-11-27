"""Final test results (run once): a table and one bar chart for README and results/.

Reads artifacts/student_vs_baselines.json (written once by `score_student --set
test`) and writes results/final_test.md, final_test.json and final_test.png. Only
summary numbers leave artifacts/; no alert, evidence or label is copied.

    python -m src.final_results

Example:
    rows = result_rows(report)   # one row per system, None where it does not apply
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from src import config

SYSTEMS = (
    ("student", "V2 student (Qwen2.5-7B + LoRA typology2)"),
    ("v1_prompt_v3", "V1 prompt v3 (gpt-4.1-mini)"),
    ("rules", "Rule baseline (settings from dev)"),
    ("always_gather_scatter", "Always GATHER_SCATTER"),
)
PANELS = (
    ("typology", "Typology right\n(visible patterns, %)"),
    ("unclear_false_alarms", '"unclear" on false alarms (%)'),
    ("invented_per_1000", "Invented facts\n(per 1,000 claims)"),
)


def count_of(measure: Dict) -> Dict:
    return {"count": measure["count"], "of": measure["of"]}


def note_writer_row(scores: Dict) -> Dict:
    """Typology, "unclear" on false alarms and invented facts of a note writer."""
    return {
        "typology": count_of(scores["typology_accuracy_visible"]),
        "unclear_false_alarms": count_of(scores["unclear_rate_false_alarms"]),
        "invented": count_of(scores["invented_facts_per_claim"]),
        "json_valid": count_of(scores["json_validity"]),
    }


def baseline_row(scores: Dict) -> Dict:
    """A baseline writes no text: no invented facts and no JSON to measure."""
    return {
        "typology": {"count": scores["typology_right_visible"],
                     "of": scores["visible"]},
        "unclear_false_alarms": {"count": scores["unclear_false_alarms"],
                                 "of": scores["false_alarms"]},
        "invented": None, "json_valid": None,
    }  # fmt: skip


def result_rows(report: Dict) -> List[Dict]:
    """One row per system, in SYSTEMS order."""
    rows = {
        "student": note_writer_row(report["student"]),
        "v1_prompt_v3": note_writer_row(report["v1_prompt_v3"]),
        "rules": baseline_row(report["rules"]),
        "always_gather_scatter": baseline_row(report["always_gather_scatter"]),
    }
    return [{"key": key, "name": name, **rows[key]} for key, name in SYSTEMS]


def cell(measure: Optional[Dict], per: int = 100) -> str:
    """ "18/37 (48.6%)", or "n/a" for a baseline that writes no text."""
    if measure is None:
        return "n/a"
    rate = per * measure["count"] / max(measure["of"], 1)
    unit = "%" if per == 100 else " per 1,000"
    return "%d/%d (%.1f%s)" % (measure["count"], measure["of"], rate, unit)


def results_markdown(rows: List[Dict]) -> str:
    """The final table as markdown (rows: systems; columns: the four measures)."""
    lines = [
        '| System | Typology right (visible patterns) | "unclear" on false alarms '
        "| Invented facts per claim | JSON valid |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append("| %s | %s | %s | %s | %s |" % (
            row["name"], cell(row["typology"]), cell(row["unclear_false_alarms"]),
            cell(row["invented"], 1000), cell(row["json_valid"]),
        ))  # fmt: skip
    return "\n".join(lines)


def panel_value(row: Dict, panel: str) -> Optional[float]:
    """The bar height of one system in one panel (None: not measured)."""
    if panel == "invented_per_1000":
        measure = row["invented"]
        return None if measure is None else 1000 * measure["count"] / measure["of"]
    return 100 * row[panel]["count"] / max(row[panel]["of"], 1)


def draw_chart(rows: List[Dict], path: Path) -> None:
    """Three small bar panels, one bar per system; "n/a" where it does not apply."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(PANELS), figsize=(12, 4.2))
    for axis, (panel, title) in zip(axes, PANELS):
        draw_panel(axis, rows, panel, title)
    figure.suptitle(
        "Final test, run once: 200 LI-Small alerts (37 visible patterns, "
        "147 false alarms)", fontsize=11,
    )  # fmt: skip
    figure.tight_layout()
    figure.savefig(path, dpi=config.RESULTS_CHART_DPI)
    plt.close(figure)


def draw_panel(axis, rows: List[Dict], panel: str, title: str) -> None:
    """One panel: the student's bar in the accent colour, the others grey."""
    labels = [r["name"].split(" (")[0] for r in rows]
    values = [panel_value(r, panel) for r in rows]
    colours = [config.RESULTS_STUDENT_COLOUR if r["key"] == "student"
               else config.RESULTS_OTHER_COLOUR for r in rows]  # fmt: skip
    positions = range(len(rows))
    axis.bar(positions, [v or 0 for v in values], color=colours)
    for position, value in zip(positions, values):
        text = "n/a" if value is None else "%.1f" % value
        axis.text(position, value or 0, text, ha="center", va="bottom", fontsize=9)
    axis.set_xticks(list(positions))
    axis.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axis.set_title(title, fontsize=10)
    axis.spines[["top", "right"]].set_visible(False)


def main() -> None:
    report = json.loads(
        (config.ARTIFACTS_DIR / config.FINAL_COMPARISON_FILE).read_text()
    )
    rows = result_rows(report)
    folder = config.RESULTS_DIR
    folder.mkdir(exist_ok=True)
    (folder / "final_test.md").write_text(results_markdown(rows) + "\n")
    (folder / "final_test.json").write_text(json.dumps(rows, indent=1) + "\n")
    draw_chart(rows, folder / "final_test.png")
    print(results_markdown(rows))


if __name__ == "__main__":
    main()
