#!/usr/bin/env python3
"""Write a matched 50-video comparison after all Gemini prompt judgments finish."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/jevany-j1s/evaluation/benchmarks/ego-gemini38/prompt50"
ROWS = [
    ("J1-S + promoted Qwen", "baselines/qwen"),
    ("J1-S + Gemini baseline", "baselines/gemini_baseline"),
    ("Event boundary", "event_boundary"),
    ("No premature absence", "no_premature_absence"),
    ("Stable action words", "stable_action_words"),
    ("Current event only", "current_event_only"),
]


def main() -> None:
    if not (OUT / "all-judging-completed").exists():
        raise ValueError("all prompt judgments must complete before reporting")
    lines = [
        "# J1-S + Gemini 3.8 Flash prompt variants on 50 EGO videos",
        "",
        "This fixed-seed random sample was drawn from the previously examined EGO split. "
        "It is a diagnostic subset, not an untouched estimate of full EGO performance. "
        "The two baseline rows reuse prior GPT-4.1 judgments on exactly these videos; "
        "the four prompt rows use new closed-loop rollouts and GPT-4.1 judgments.",
        "",
        "GPT-4.1 `pauc-f-v1`, `omega=0.5`; `R` is the PAUC recall component.",
        "",
        "| Selector + generator prompt | P | R | PAUC-F | Fires | Useful | Wasted | Fires/video |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, relative in ROWS:
        summary = json.loads((OUT / relative / "pauc_f_summary-gpt-4.1-v1.json").read_text())
        if summary["num_videos"] != 50:
            raise ValueError(f"incomplete score for {label}")
        lines.append(
            f"| {label} | {summary['fire_precision']:.2f}% | "
            f"{summary['pauc_recall_component']:.2f} | "
            f"{summary['pauc_f']:.2f} | {summary['num_fires']:,} | "
            f"{summary['num_value_adding_fires']:,} | {summary['num_wasted_fires']:,} | "
            f"{summary['num_fires']/50:.2f} |"
        )
    lines += [
        "",
        "Selection IDs and stratum counts are in [selection_audit.json](selection_audit.json) "
        "and [selected_ids.json](selected_ids.json). Each variant directory contains its "
        "prompt text, per-video predictions, judge outputs, and PAUC-F summary.",
        "",
    ]
    comparison = "\n".join(lines)
    (OUT / "comparison.md").write_text(comparison)
    note = ROOT / "docs/exp/jevany_j1s_gemini38_prompt50.md"
    current = note.read_text()
    marker = "## Results\n\n"
    if marker not in current:
        raise ValueError("experiment note has no Results section")
    table_start = comparison.index("| Selector + generator prompt |")
    table_end = comparison.index("\n\n", table_start)
    table = comparison[table_start:table_end]
    note.write_text(current.split(marker, 1)[0] + marker + table + "\n")
    print(OUT / "comparison.md")


if __name__ == "__main__":
    main()
