#!/usr/bin/env python3
"""Write no-judge P/R/PAUC-F and response-rate table for J1-S-R v2."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/jevany-j1sr-v2"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def row(name: str, source: dict, *, validation: bool = False) -> dict:
    videos = source["num_sampled_videos"] if validation else source["num_videos"]
    fires = source["total_physical_fires"] if validation else source["num_fires"]
    return {"controller": name, "videos": videos, "responses": fires,
            "responses_per_video": fires / videos,
            "precision": source["fire_precision"],
            "pauc_recall": source["pauc_recall_component"],
            "pauc_f1": source["pauc_f"]}


def main() -> None:
    validation = OUT / "evaluation/validation"
    scores = {
        "validation": [
            row("J1-S", read(validation / "j1s/official/results/timing_prejudge.json"), validation=True),
            row("J1-S-R", read(validation / "j1sr/official/results/timing_prejudge.json"), validation=True),
        ]
    }
    for split in ("web", "ego"):
        folder = OUT / f"evaluation/benchmarks/{split}"
        scores[split] = [
            row("J1-S", read(OUT / f"evaluation/baseline_timing/{split}/pauc_f_summary-timing-v1.json")),
            row("J1-S-R", read(folder / "pauc_f_summary-timing-v1.json")),
        ]
    dest = OUT / "reports"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "timing.json").write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n")
    lines = ["# J1-S-R v2 timing-only results", "",
             "Answer quality is held at maximum for in-window fires. These are diagnostics, not judged scores.", "",
             "| Split | Controller | P | R (PAUC) | F1 (PAUC-F) | Responses/video |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for split, rows in scores.items():
        for item in rows:
            lines.append(f"| {split.upper()} | {item['controller']} | {item['precision']:.2f}% | "
                         f"{item['pauc_recall']:.2f}% | {item['pauc_f1']:.2f}% | "
                         f"{item['responses_per_video']:.3f} |")
    (dest / "timing.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(scores, sort_keys=True))


if __name__ == "__main__":
    main()
